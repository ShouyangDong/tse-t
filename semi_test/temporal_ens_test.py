import numpy as np
import cv2
import os
import torch
import torchvision
from maskrcnn_benchmark.modeling.roi_heads.box_head.inference import PostProcessor
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
import glob
str_folder = './tempor_pred_save/'
str_img = './VOC2007/JPEGImages/'
str_output = './output'

def box_soft_nms(bboxes,scores,labels,nms_threshold=0.3,soft_threshold=0.3,sigma=0.5,mode='union'):
    """
    soft-nms implentation according the soft-nms paper
    :param bboxes:
    :param scores:
    :param labels:
    :param nms_threshold:
    :param soft_threshold:
    :return:
    """
    unique_labels = labels.cpu().unique().cuda()

    box_keep = []
    labels_keep = []
    scores_keep = []
    for c in unique_labels:
        c_boxes = bboxes[labels == c]
        c_scores = scores[labels == c]
        weights = c_scores.clone()
        x1 = c_boxes[:, 0]
        y1 = c_boxes[:, 1]
        x2 = c_boxes[:, 2]
        y2 = c_boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        _, order = weights.sort(0, descending=True)
        while order.numel() > 0:
            #print(order.numel(),len(box_keep),order.numel())
            if order.numel() > 1:
                i = order[0]
            else:
                i = order

            box_keep.append(c_boxes[i])
            labels_keep.append(c)
            scores_keep.append(c_scores[i])

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1 + 1).clamp(min=0)
            h = (yy2 - yy1 + 1).clamp(min=0)
            inter = w * h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids_t= (ovr>=nms_threshold).nonzero().squeeze()

            weights[[order[ids_t+1]]] *= torch.exp(-(ovr[ids_t] * ovr[ids_t]) / sigma)

            ids = (weights[order[1:]] >= soft_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            c_boxes = c_boxes[order[1:]][ids]
            c_scores = weights[order[1:]][ids]
            _, order = weights[order[1:]][ids].sort(0, descending=True)
            if c_boxes.dim()==1:
                c_boxes=c_boxes.unsqueeze(0)
                c_scores=c_scores.unsqueeze(0)
            x1 = c_boxes[:, 0]
            y1 = c_boxes[:, 1]
            x2 = c_boxes[:, 2]
            y2 = c_boxes[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    return box_keep, labels_keep, scores_keep

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    scores = predictions.get_field('scores')
    boxes = predictions.bbox
    
    #boxes,labels,scores = box_soft_nms(predictions.bbox,predictions.get_field('objectness'),predictions.get_field("labels"))
    # keep = torchvision.ops.nms(predictions.bbox,predictions.get_field('scores'), 0.5)
    # boxes,labels,scores = boxes[keep],labels[keep],scores[keep]
    mask = scores > 0.9
    boxes,labels,scores = boxes[mask],labels[mask],scores[mask]


    #print(predictions.get_field('objectness')[:10])

    for box  in boxes:
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        print(top_left,bottom_right)
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), (0,0,255), 1
        )

    return image


temporal_ens = glob.glob(str_folder+'**/*.pt',recursive=True)

postprocessor = PostProcessor(
    score_thresh = 0.05,
    nms = 0.5,
    detections_per_img = 100,
    box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0)),
    cls_agnostic_bbox_reg = False,
    bbox_aug_enabled = False
)


for _iter in temporal_ens:
    str_file = os.path.basename(_iter).split('_x')[0]
    _id = str_file.replace('_iter','')
    boxes = torch.load(_iter)

    boxes_nms = postprocessor.filter_results(boxes,21)

    str_id = _id + '.jpg'
    print(str_img + str_id)
    # if(str_id.find('009726')) < 0:
    #     continue
    image = cv2.imread(str_img + str_id)
    img = overlay_boxes(image, boxes_nms)
    cv2.imwrite(str_output+str_id,img)

print('process_end')