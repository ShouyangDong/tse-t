# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from apex import amp
import numpy as np
from maskrcnn_benchmark.modeling.rpn.inference import RPNPostProcessor
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
from maskrcnn_benchmark.modeling.utils import cat 
from maskrcnn_benchmark import _C
nms = amp.float_function(_C.soft_nms)

_box_nms = nms

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    return layer

def boxlist_nms(boxlist, nms_thresh, max_proposals=-1,siagma=1.2, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep,score_soft = _box_nms(boxes, score, nms_thresh,siagma)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    boxlist.add_field('scores',score_soft)
    #print(score[keep],score_soft)
    return boxlist.convert(mode)


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result



class PostProcessor_retina(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def one_pt_scores(pts_sorted,postprocessor,ncls = 21):
    sel_pt = pts_sorted[0]
    bbox = torch.load(sel_pt)
    # labels = sel_pt.get_field("labels")
    # scores = sel_pt.get_field('scores')
    # boxes = sel_pt.bbox

    boxes_nms = postprocessor.filter_results(bbox,ncls)
    return boxes_nms


def multi_pt_scores(pts_sorted,postprocessor,ncls = 21):
    bboxes = []
    scores = []

    bbox_info = torch.load(pts_sorted[0])
    for _pt in pts_sorted:
        bbox = torch.load(_pt)
        bboxes.append(bbox.bbox.reshape([-1,ncls,4]))
        scores.append(bbox.get_field('scores').reshape([-1,ncls]))

    boxes_comb = BoxList(torch.cat(bboxes).reshape([-1,4]),bbox_info.size)
    boxes_comb.add_field('scores',torch.cat(scores).reshape([-1]))
    boxes_nms = postprocessor.filter_results(boxes_comb,ncls)
    return boxes_nms

def sparse_to_dense(bbox,scores,s_ind,total_len):
    n_cls = scores.shape[1]
    bbox_dense = torch.zeros([total_len,4],dtype=bbox.dtype).to(bbox.device)
    scores_dense = torch.ones([total_len,n_cls],dtype=scores.dtype).to(bbox.device)*(-10.)
    if len(s_ind) > 0:
        bbox_dense[s_ind] = bbox
        scores_dense[s_ind] = scores
    return bbox_dense,scores_dense

# def sparse_to_dense_wrap(bbox):
#     bbox_dense,scores_dense = sparse_to_dense(bbox.bbox,bbox.get_field('scores'),bbox.get_field('sel_ind'),bbox.get_field('total_ind_len'))
#     bbox_out = BoxList(bbox_dense, bbox.size, mode="xyxy")
#     bbox_out.extra_fields['scores'] = scores_dense
#     bbox_out.extra_fields['feat_w_h'] = scores_dense
#     return bbox_out
    

def check_anchors(anchors,feat_sz_hw,image_hw_sz,anchors_len = 9):
    shifts_x = torch.arange(
        0, feat_sz_hw[1], dtype=torch.float32, device=anchors.device
    )
    shifts_y = torch.arange(
        0, feat_sz_hw[0], dtype=torch.float32, device=anchors.device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    pos_rate_y = shift_y / feat_sz_hw[0]
    pos_rate_x = shift_x / feat_sz_hw[1]

    x_min,x_max = anchors.view([-1,4])[:,0]/image_hw_sz[1],anchors.view([-1,4])[:,2]/image_hw_sz[1] 
    y_min,y_max = anchors.view([-1,4])[:,1]/image_hw_sz[0],anchors.view([-1,4])[:,3]/image_hw_sz[0]

    x_min = x_min.view([-1,9])
    y_min = y_min.view([-1,9])

    x_max = x_max.view([-1,9])
    y_max = y_max.view([-1,9])

    for i in range(anchors_len):
        _x = (pos_rate_x.view(-1) >  x_min[:,i]) & (pos_rate_x.view(-1) <  x_max[:,i])
        _y =  (pos_rate_y.view(-1) >  y_min[:,i]) & (pos_rate_y.view(-1) <  y_max[:,i])
        assert _x.all()
        assert _y.all()



def check_resuts(resut,anchors_len = 9):
    bbox_len = resut.bbox.shape[1]
    n_cls = resut.get_field('scores').shape[1]

    total_len = resut.get_field('total_ind_len')
    shapes_wh = resut.get_field('feat_w_h')
    W,H = int(shapes_wh[0]),int(shapes_wh[1])
    if(len(resut.get_field('sel_ind'))) == 0:
        return
    shape_wh_zeros = shapes_wh
    bbox_dense,scores_dense = sparse_to_dense(resut.bbox,resut.get_field('scores'),resut.get_field('sel_ind'),total_len)
    
    bbox_hw = bbox_dense.view(H,W,anchors_len,4)
    scores_hw = scores_dense.view(H,W,anchors_len,n_cls)

    scors_hw_max = torch.argmax(torch.sigmoid(scores_hw))
    rat_hw_pos = np.unravel_index(scors_hw_max.cpu(),scores_hw.shape)
    H_rate = rat_hw_pos[0]*1./float(H) 
    W_rate = rat_hw_pos[1]*1./float(W)
    predresutbox = bbox_hw[rat_hw_pos[0]][rat_hw_pos[1]][rat_hw_pos[2]] 

    #-----------------------------
    restore_idx = np.array(np.unravel_index(resut.get_field('sel_ind').cpu(),[H,W,anchors_len])).T
    img_hw = resut.size[::-1]
    hw_rate = np.vstack([restore_idx[:,0]*1./H,restore_idx[:,1]*1./W]).T

    cent_yx =  hw_rate*np.array(img_hw)
    # print('yyyy',np.logical_and(cent_yx[:,0]  > resut.bbox[:,1].numpy() ,  cent_yx[:,0] < resut.bbox[:,3].numpy()).any())
    # print('xxxx',np.logical_and(cent_yx[:,1]  > resut.bbox[:,0].numpy() ,  cent_yx[:,1] < resut.bbox[:,2].numpy()).any())
    assert np.logical_and(cent_yx[:,0]  > resut.bbox[:,1].cpu().numpy() ,  cent_yx[:,0] < resut.bbox[:,3].cpu().numpy()).any()
    assert np.logical_and(cent_yx[:,1]  > resut.bbox[:,0].cpu().numpy() ,  cent_yx[:,1] < resut.bbox[:,2].cpu().numpy()).any()


    #---------------------------


    # dict_z = {'feat_w_h':[W,H],'max_val':torch.max(torch.sigmoid(scores_hw)),'max_pos_hw':rat_hw_pos,
    # 'relative_pos_feat_wh':[W_rate*float(shape_wh_zeros[0]),H_rate*float(shape_wh_zeros[1])],
    # 'relative_pos_img_wh':[W_rate*float(resut.size[0]),H_rate*float(resut.size[1])],
    # 'predict_pos':pred_bbox}
    #print(dict_z)
    return None



def checks(pts_sorted,postprocessor,anchors_len = 9):
    bboxes = []
    scores = []

    bbox_info = torch.load(pts_sorted[0])
    bbox_ts = [torch.load(_pt) for _pt in pts_sorted]
    bbox_layers = list(zip(*bbox_ts))

    for b_l in bbox_layers:
        # total_len = [_b_l.get_field('total_ind_len') for _b_l in b_l]
        # shapes = [_b_l.get_field('feat_w_h') for _b_l in b_l]
        # z_len = torch.stack(total_len)

        shape_wh_zeros = None
        for _b in b_l:
            check_resuts(_b,anchors_len)

        #print('---------------------------------------------------------')    
            #F.interpolate(last_inner, scale_factor=2, mode="nearest")

    return None


def ens_teacher(bbox_ts,img_sz,anchor_strides,anchors_len = 9,sel_threshold=0.05):
    #sel_threshold=0.01
    #bbox_info = bbox_ts[0]
    bbox_info = bbox_ts[0]
    n_cls = bbox_info[0].get_field('scores').shape[1]
    img_sz_real = bbox_info[0].size
    
    bboxes_ens = []
    scores_ens = []
    sel_ens = []
    feat_w_h_ens = []
    bbox_layers = list(zip(*bbox_ts))
    for i,b_l in enumerate(bbox_layers):
        shape_wh_zeros = None
        #feat_wh_0 = bbox_info[i].get_field('feat_w_h')
        #for resize ,feat_wh not similar
        H_real = (img_sz[0] + anchor_strides[i] - 1)//anchor_strides[i]
        W_real = (img_sz[1] + anchor_strides[i] - 1)//anchor_strides[i]
        feat_wh_0 = torch.tensor([W_real,H_real],dtype=torch.int32)
        bboxs = []
        scores = []
        for _b in b_l:
            # try:
            #     check_resuts(_b,anchors_len)
            # except Exception as e:
            #     print('error file',pts_sorted[0])
            #     continue
            feat_wh = _b.get_field('feat_w_h')

            if len(_b.get_field('sel_ind')) > 0:
                assert _b.get_field('sel_ind').max() < _b.get_field('total_ind_len')
            bbox_dense,scores_dense = sparse_to_dense(_b.bbox,_b.get_field('scores'),_b.get_field('sel_ind'),_b.get_field('total_ind_len'))
            _bbox = bbox_dense.view(1,feat_wh[1],feat_wh[0],anchors_len*4).permute(0,3,1,2)
            _scores = scores_dense.view(1,feat_wh[1],feat_wh[0],anchors_len*n_cls).permute(0,3,1,2)
            # method 1 for resize
            _bbox_resize = F.interpolate(_bbox, size = [feat_wh_0[1],feat_wh_0[0]], mode="nearest").permute(0,2,3,1).view(feat_wh_0[1],feat_wh_0[0],anchors_len,4)
            _scores_resize = F.interpolate(_scores, size = [feat_wh_0[1],feat_wh_0[0]], mode="nearest").permute(0,2,3,1).view(feat_wh_0[1],feat_wh_0[0],anchors_len,n_cls)
            bboxs.append(_bbox_resize)
            scores.append(_scores_resize)
            # method 2 for memory
            # assert feat_wh_0[0] == feat_wh[0] and feat_wh_0[1] == feat_wh[1]
            # _bbox_resize = _bbox.permute(0,2,3,1).view(feat_wh_0[1],feat_wh_0[0],anchors_len,4)
            # _scores_resize = _scores.permute(0,2,3,1).view(feat_wh_0[1],feat_wh_0[0],anchors_len,n_cls)
            # bboxs.append(_bbox_resize)
            # scores.append(_scores_resize)

        scores_stack = torch.stack(scores,dim=0)
        scores_means = torch.mean(scores_stack.sigmoid(),dim=0)
        scores_means_sigmoid = scores_means.view(-1,n_cls)

        #weighted mean for bbox
        bboxes_stack = torch.stack(bboxs,dim=0)
        scores_bboxes = scores_stack.sigmoid().max(4).values
        #scores_bboxes[scores_bboxes < 0.01] = 0 #??  index 16777275 is out of bounds for dimension 1 with size 100
        scores_bboxes = scores_bboxes*(scores_bboxes > 0.01).to(scores_bboxes.dtype)
        bboxes_weight = scores_bboxes/(torch.sum(scores_bboxes,dim=0,keepdim=True) + 1e-4)
        bboxes_weight_means = torch.sum(bboxes_stack * bboxes_weight[:,:,:,:,None],dim=0).view(-1,4)

        #############################################
        scores_means_sel = (scores_means_sigmoid.max(1).values > sel_threshold).nonzero().view(-1)

        bboxes_ens.append(bboxes_weight_means[scores_means_sel])
        scores_ens.append(scores_means_sigmoid[scores_means_sel])
        feat_w_h_ens.append(feat_wh_0)
        sel_ens.append(scores_means_sel)
    return {'ens_boxes':bboxes_ens,'ens_scores':scores_ens,'feat_w_h':feat_w_h_ens,'image_size':img_sz_real,'sel_ind':sel_ens}

def concat_layers_nms(bbox_ts,postprocessor):
    pred_bbox = []
    for bbox_img in bbox_ts:
        bboxes_ens = cat([_bbox.bbox for _bbox in bbox_img],dim=0)
        if len(bboxes_ens) < 1:
            img_tmp = BoxList(torch.zeros([0,4]),[400,400])
            img_tmp.add_field('sel_ind',[])
            img_tmp.add_field('scores',torch.zeros([0,]))
            img_tmp.add_field('labels',torch.zeros([0,]))
            pred_bbox.append(img_tmp)
            continue
        scores_means_sigmoid = cat([_bbox.get_field('ens_scores') for _bbox in bbox_img],dim=0)
        sel_ens = cat([_bbox.get_field('sel_ind') for _bbox in bbox_img],dim=0)

        #############################################
        scores_means_sel = scores_means_sigmoid > 0.05
        per_candidate_nonzeros = torch.nonzero(scores_means_sel)
        box_loc = per_candidate_nonzeros[:, 0]
        per_scores = scores_means_sigmoid[scores_means_sel]
        per_class = per_candidate_nonzeros[:, 1]
        per_class += 1        

        assert bbox_img[0].size != [400,400]        #exception error post_process img_tmp = BoxList(torch.zeros([0,4]),[400,400]), voc.py img_tmp = BoxList(torch.zeros([0,4]),[400,400])
        boxes_comb = BoxList(bboxes_ens[box_loc],bbox_img[0].size)
        boxes_comb = boxes_comb.clip_to_image(remove_empty=False)
        boxes_comb.add_field('scores',per_scores)
        boxes_comb.add_field('labels',per_class)


        boxes_nms = postprocessor.select_over_all_levels([boxes_comb])
        pred_bbox.append(boxes_nms[0])
    return pred_bbox


def multi_align_ens(pts_sorted,postprocessor,anchors_len = 9):
    pts_sorted = pts_sorted[-8:]
    bbox_info = torch.load(pts_sorted[0])
    bbox_ts = []
    for  _pt in pts_sorted:
        _bb = torch.load(_pt)
        bbox_ts.append(_bb)
    return multi_align_ens_bbox(bbox_ts,postprocessor,anchors_len)


def multi_align_ens_bbox(bbox_ts,postprocessor,anchors_len = 9):
    bboxes_ens = []
    scores_ens = []
    labels_ens = []


    bbox_layers = list(zip(*bbox_ts))
    n_cls = bbox_ts[0][0].get_field('scores').shape[1]

    for i,b_l in enumerate(bbox_layers):
        # total_len = [_b_l.get_field('total_ind_len') for _b_l in b_l]
        # shapes = [_b_l.get_field('feat_w_h') for _b_l in b_l]
        # z_len = torch.stack(total_len)

        shape_wh_zeros = None
        feat_wh_0 = bbox_ts[0][i].get_field('feat_w_h')
        bboxs = []
        scores = []
        for _b in b_l:
            # try:
            #     check_resuts(_b,anchors_len)
            # except Exception as e:
            #     print('error file',pts_sorted[0])
            #     continue
            feat_wh = _b.get_field('feat_w_h')
            bbox_dense,scores_dense = sparse_to_dense(_b.bbox,_b.get_field('scores'),_b.get_field('sel_ind'),_b.get_field('total_ind_len'))
            _bbox = bbox_dense.view(1,feat_wh[1],feat_wh[0],anchors_len*4).permute(0,3,1,2)
            _scores = scores_dense.view(1,feat_wh[1],feat_wh[0],anchors_len*n_cls).permute(0,3,1,2)
            _bbox_resize = F.interpolate(_bbox, size = [feat_wh_0[1],feat_wh_0[0]], mode="nearest").permute(0,2,3,1).view(feat_wh_0[1],feat_wh_0[0],anchors_len,4)
            _scores_resize = F.interpolate(_scores, size = [feat_wh_0[1],feat_wh_0[0]], mode="nearest").permute(0,2,3,1).view(feat_wh_0[1],feat_wh_0[0],anchors_len,n_cls)
            bboxs.append(_bbox_resize)
            scores.append(_scores_resize)

        scores_stack = torch.stack(scores,dim=0)
        scores_means = torch.mean(scores_stack,dim=0)
        scores_means_sigmoid = scores_means.sigmoid().view(-1,n_cls)

        #weighted mean for bbox
        bboxes_stack = torch.stack(bboxs,dim=0)
        scores_bboxes = scores_stack.sigmoid().max(4).values
        scores_bboxes[scores_bboxes < 0.01] = 0
        bboxes_weight = scores_bboxes/(torch.sum(scores_bboxes,dim=0,keepdim=True) + 1e-4)
        bboxes_weight_means = torch.sum(bboxes_stack * bboxes_weight[:,:,:,:,None],dim=0).view(-1,4)

        #############################################
        scores_means_sel = scores_means_sigmoid > 0.05
        per_candidate_nonzeros = torch.nonzero(scores_means_sel)
        box_loc = per_candidate_nonzeros[:, 0]
        per_scores = scores_means_sigmoid[scores_means_sel]
        per_class = per_candidate_nonzeros[:, 1]
        per_class += 1


        bboxes_ens.append(bboxes_weight_means[box_loc])
        scores_ens.append(per_scores)
        labels_ens.append(per_class)

    boxes_comb = BoxList(torch.cat(bboxes_ens).reshape([-1,4]),bbox_ts[0][0].size)
    boxes_comb = boxes_comb.clip_to_image(remove_empty=False)
    boxes_comb.add_field('scores',torch.cat(scores_ens))
    boxes_comb.add_field('labels',torch.cat(labels_ens))

    boxes_nms = postprocessor.select_over_all_levels([boxes_comb])
    return boxes_nms



class RetinaNetPostProcessorSemi(RPNPostProcessor):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        retina_anchor_strides,
        box_coder=None,
        sel_thr = 0.05
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(RetinaNetPostProcessorSemi, self).__init__(
            pre_nms_thresh, 0, nms_thresh, min_size
        )
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.retina_anchor_strides = retina_anchor_strides
        self.sel_thr = sel_thr

        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
 
    def add_gt_proposals(self, proposals, targets):
        """
        This function is not used in RetinaNet
        """
        pass
    def forward(self, anchors, objectness, box_regression,image_size, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b,stride in zip(anchors, objectness, box_regression,self.retina_anchor_strides):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b,stride,image_size))

        boxlists = list(zip(*sampled_boxes))
        #boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        return boxlists

    def forward_for_single_feature_map(
            self, anchors, box_cls, box_regression,stride,image_size):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = box_cls.device
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

        num_anchors = A * H * W

        results = []
        for per_box_cls, per_box_regression, per_anchors,im_sz_hw in zip(
            box_cls,
            box_regression,
            anchors,image_size):

            detections = self.box_coder.decode(
                per_box_regression.reshape(-1, 4),
                per_anchors.bbox.view(-1, 4)
            )



            H_real = (im_sz_hw[0] + stride - 1)//stride
            W_real = (im_sz_hw[1] + stride - 1)//stride
            detections = detections.reshape(H,W,A,4)
            per_box_cls_pad_cut = per_box_cls[:H_real,:W_real,:,:].reshape(-1,C)
            detections_pad_cut = detections[:H_real,:W_real,:,:].reshape(-1,4)


            #-------------------------------debug check
            # scors_hw_max = torch.argmax(torch.sigmoid(per_box_cls_pad_cut))
            # rat_hw_pos = np.unravel_index(int(scors_hw_max),[H_real,W_real,A, C])
            # prob_ind = np.unravel_index(int(scors_hw_max),list(per_box_cls.shape))
            # if torch.sigmoid(per_box_cls[prob_ind]) > 0.5:
            #     cent_x,cent_y =  anchors[0].size[0]*rat_hw_pos[1]*1./W,anchors[0].size[1]*rat_hw_pos[0]*1./H
            #     assert cent_x  > detections[prob_ind[0]][0] and  cent_x  < detections[prob_ind[0]][2]
            #     assert cent_y  > detections[prob_ind[0]][1] and  cent_y  < detections[prob_ind[0]][3]

                #print('heat feat, HW: {} {},loc pos:{}, scores:{}, decode bbox {} ,anchors {}'.format(H,W,rat_hw_pos,torch.sigmoid(per_box_cls[prob_ind]),\
                #                                                                                detections[prob_ind[0]],per_anchors.bbox.view(-1, 4)[prob_ind[0]]))
            #-------------------------------debug check end

            sel_ind = torch.nonzero(torch.max(per_box_cls_pad_cut.sigmoid(),dim=1).values > self.sel_thr).squeeze(1)
            boxlist = BoxList(detections_pad_cut[sel_ind], per_anchors.size, mode="xyxy")
            boxlist.add_field("scores", per_box_cls_pad_cut[sel_ind])
            boxlist.add_field("sel_ind", sel_ind.to(boxlist.bbox.device))
            boxlist.add_field("total_ind_len", torch.tensor(len(detections_pad_cut)).view(1))
            boxlist.add_field("feat_w_h",torch.tensor([W_real,H_real]))
            assert len(detections_pad_cut) == 9*(W_real*H_real)
            

            results.append(boxlist)

        return results

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms_thresh,
                    score_field="scores"
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_retinanet_semi_postprocessor(config, rpn_box_coder, is_train):
    pre_nms_thresh = config.MODEL.RETINANET.INFERENCE_TH
    pre_nms_top_n = config.MODEL.RETINANET.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.RETINANET.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    retina_anchor_strides = config.MODEL.RETINANET.ANCHOR_STRIDES
    sel_thr = config.SEMI.SEL_THR
    min_size = 0

    box_selector = RetinaNetPostProcessorSemi(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=min_size,
        num_classes=config.MODEL.RETINANET.NUM_CLASSES,
        retina_anchor_strides=retina_anchor_strides,
        box_coder=rpn_box_coder,
        sel_thr = sel_thr
    )

    return box_selector
