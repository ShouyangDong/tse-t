# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from .post_process import permute_and_flatten

class SemiLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets


    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals


    def __call__(self, proposals, targets):
        
        device = proposals[0].bbox.device
        classification_loss = []
        box_loss = []
        proposals = self.subsample(proposals, targets)


        for _proposal in proposals:
            
            _class_logits = _proposal.get_field('class_logits')
            _box_regression = _proposal.get_field('box_regression')
            labels = _proposal.get_field("labels")
            regression_targets = _proposal.get_field("regression_targets")

            _classification_loss = F.cross_entropy(_class_logits, labels)/len(proposals)

            # get indices that correspond to the regression targets for
            # the corresponding ground truth labels, to be used with
            # advanced indexing
            sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
            labels_pos = labels[sampled_pos_inds_subset]
            if self.cls_agnostic_bbox_reg:
                map_inds = torch.tensor([4, 5, 6, 7], device=device)
            else:
                map_inds = 4 * labels_pos[:, None] + torch.tensor(
                    [0, 1, 2, 3], device=device)

            _box_loss = smooth_l1_loss(
                _box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                size_average=False,
                beta=1,
            )
            _box_loss = _box_loss / labels.numel() / len(proposals)
            classification_loss.append(_classification_loss)
            box_loss.append(_box_loss)


        return {'semi_cls':classification_loss,'semi_box_reg':box_loss}

        

    def call__b(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


class SemiLossComputation_ext(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        return matched_idxs

    def prepare_targets(self, proposals, targets):
        losses = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_idxs = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs_target = matched_idxs[matched_idxs > -1]
            proposals_per_image = proposals_per_image[matched_idxs > -1]
            targets_per_image = targets_per_image[matched_idxs_target]
            
            proposals_per_image.add_field('class_logits',(F.one_hot(proposals_per_image.get_field('labels'),81).float().T*proposals_per_image.get_field('scores')).T)
            targets_per_image.add_field('class_logits',F.one_hot(targets_per_image.get_field('labels'),81).float())            

            matched_idx = proposals_per_image.get_field('labels') == targets_per_image.get_field('labels')
            reg_loss = F.mse_loss(proposals_per_image.bbox[matched_idx],targets_per_image.bbox[matched_idx])
            cls_loss = F.mse_loss(proposals_per_image.get_field('class_logits'),targets_per_image.get_field('class_logits'),reduction='sum')
            losses.append({'reg_loss':reg_loss,'cls_loss':cls_loss,})

        return losses


    def __call__(self, proposals, targets):
        
        device = proposals[0].bbox.device
        classification_loss = []
        box_loss = []
        losses = self.prepare_targets(proposals, targets)
        return losses


def make_semi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.SEMI.FG_IOU_THRESHOLD,
        cfg.SEMI.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )


    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = SemiLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator

# TODO maybe push this to nn?
def smooth_l1_loss_ext(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    thr = (1/3*(beta**3) - beta)/ (-beta) 
    #loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    loss = torch.where(cond, (1/3) * n ** 3 / beta, n - thr * beta)
    if size_average:
        return loss.mean()
    return loss.sum(1)


def focal_loss(input,target,gamma = 0):
    loss = -torch.abs(input - target)**gamma*target*torch.log(input)
    return loss

def semi_loss_fn(predict,predict_ema,ensembles,img_sz,box_coder,strides=(8,16,32,64,128),anchor=9,n_cls=20,reg_cons_w=20):
    unpredict_sample = [len(_ens)!=len(predict['rpn_cls']) for _ens in ensembles]
    if all(unpredict_sample):
        return {'cls_consistence':torch.tensor(0,dtype=torch.float32),'reg_consistence':torch.tensor(0,dtype=torch.float32)}

    predict_cls_layers = predict['rpn_cls']
    predict_cls_layers_ema = predict_ema['rpn_cls']
    predict_bbox_layers = predict['rpn_box_encode']
    anchors_layers  = list(zip(*predict['anchors']))
    ensembles_layers = list(zip(*ensembles))


    losses_cls = []
    losses_reg = []
    sel_sum = 0
    for i,(pred_cls_layer,pred_cls_layer_ema,pred_bbox_layer,anchors_layer,ens_layers,stride) in enumerate(zip(predict_cls_layers,predict_cls_layers_ema,predict_bbox_layers,anchors_layers,ensembles_layers,strides)):
        pred_clses = pred_cls_layer.split(1)
        pred_clses_ema = pred_cls_layer_ema.split(1)

        ens_bach_cls = []
        ens_bach_reg = []

        for _id,(_cls,_cls_ema,_bbox,_anchor,_ens) in enumerate(zip(pred_clses,pred_clses_ema,pred_bbox_layer,anchors_layer,ens_layers)):
            if len(_ens.get_field('sel_ind')) < 1:
                ens_bach_cls.append(torch.tensor(0.).to(_cls.device).view(-1))
                ens_bach_reg.append(torch.tensor(0.).to(_cls.device).view(-1))
                continue
            H_real = (img_sz[_id][0] + stride - 1)//stride
            W_real = (img_sz[_id][1] + stride - 1)//stride
            assert W_real == _ens.get_field('feat_w_h')[0]
            assert H_real == _ens.get_field('feat_w_h')[1]
            _cls_real = _cls[:,:,:H_real,:W_real]
            _cls_align = permute_and_flatten(_cls_real,1, anchor, n_cls, H_real, W_real).reshape([-1,n_cls])
            _cls_sel = _cls_align[_ens.get_field('sel_ind')]

            _cls_real_ema = _cls_ema[:,:,:H_real,:W_real]
            _cls_align_ema = permute_and_flatten(_cls_real_ema,1, anchor, n_cls, H_real, W_real).reshape([-1,n_cls])
            _cls_sel_ema = _cls_align_ema[_ens.get_field('sel_ind')]


            sel_sum += len(_ens.get_field('sel_ind'))
            with torch.no_grad():
                teacher_sigmod = (_ens.get_field('ens_scores')*0.8 + _cls_sel_ema.sigmoid()*0.2)
            #---------------cls loss
            # cls_mse_loss = F.mse_loss(_cls_sel.sigmoid(), teacher_sigmod, reduction='none')
            # cls_mse_loss[cls_mse_loss < 0.003] = 0
            # ens_teacher_cls_loss = cls_mse_loss.sum().view(-1)#F.mse_loss(_cls_sel.sigmoid(), teacher_sigmod, reduction='sum').view(-1)
            # ens_bach_cls.append(ens_teacher_cls_loss)
            #---------------reg loss
            H_padding,W_padding = _bbox.shape[1],_bbox.shape[2]
            _anchor_align = _anchor.bbox.view(H_padding,W_padding,anchor,4)[:H_real,:W_real,:,:].reshape(-1,4)
            _bbox_real = _bbox[:,:H_real,:W_real]
            _bbox_align = permute_and_flatten(_bbox_real,1, anchor, 4, H_real, W_real).reshape([-1,4])
            _bbox_sel = _bbox_align[_ens.get_field('sel_ind')]
            _anchor_sel = _anchor_align[_ens.get_field('sel_ind')]
            _ens_encode = box_coder.encode(_ens.bbox,_anchor_sel)

            # reg_mse_loss = F.mse_loss(_bbox_sel, _ens_encode,reduction='none').view(-1)
            # reg_mse_loss[reg_mse_loss < 0.008] = 0
            # ens_teacher_reg_loss = reg_mse_loss.sum().view(-1)#F.mse_loss(_bbox_sel, _ens_encode,reduction='sum').view(-1)
            # ens_bach_reg.append(ens_teacher_reg_loss)
            #new losses
            bbox_loss_sum = smooth_l1_loss_ext(_bbox_sel, _ens_encode,beta=0.4,size_average = False)/4
            with torch.no_grad():
                bbox_weight = teacher_sigmod.max(1).values
                bbox_weight = bbox_weight ** 0.5
            new_bbox_loss = bbox_loss_sum * bbox_weight
            new_cls_loss = focal_loss(_cls_sel.sigmoid(), teacher_sigmod,gamma=2)
            ens_bach_cls.append(new_cls_loss.sum().view(-1))
            ens_bach_reg.append(new_bbox_loss.sum().view(-1))



        losses_cls.append(torch.cat(ens_bach_cls))
        losses_reg.append(torch.cat(ens_bach_reg))

    return {'cls_consistence':torch.stack(losses_cls).sum(0),'reg_consistence':torch.stack(losses_reg).sum(0)/(reg_cons_w + 1e-3)}
