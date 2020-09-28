import numpy as np
import cv2
import os
import argparse
import torch
import torchvision
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.inference import make_retinanet_postprocessor
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.structures.bounding_box import BoxList
from tqdm import tqdm
import glob
import json
import multiprocessing as mp
#nms
from maskrcnn_benchmark.modeling.roi_heads.box_head.inference import PostProcessor
#soft nms
#from semi_test.post_process import PostProcessor
from semi_test.post_process import multi_pt_scores,one_pt_scores,checks,multi_align_ens
mp.set_start_method('spawn', True)


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


def mAP(cfg,predictions):
    box_only = False
    iou_types = ('bbox',)
    expected_results = []
    expected_results_sigma_tol = 4
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)

    mAps = []
    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    for _data in data_loaders_val:
        ids_sorted_by_db = _data.dataset.ids
        predict_db = [predictions[_iter] for _iter in ids_sorted_by_db]
        mAp_scores = evaluate(dataset=_data.dataset,
                        predictions=predict_db,
                        output_folder='./',
                        **extra_args)
        mAps.append(mAp_scores)

    return mAps



def multi_process(pts_folder,postprocessor):
    pts = glob.glob(os.path.join(pts_folder,'*.pt'))
    img_id = os.path.basename(pts_folder)
    pts_iter = [int(_id.split('_x')[-1].replace('.pt','')) for _id in pts]
    idx_sorted = np.argsort(pts_iter)
    pts_sorted = np.array(pts)[idx_sorted]
    bbox = multi_align_ens(pts_sorted,postprocessor)
    return img_id, bbox

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    str_folder = './tempor_pred_coco_bn8/'
    str_img = '/JPEGImages/'
    str_output = 'output/'

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()




    # postprocessor = PostProcessor_retina(
    #     score_thresh = 0.05,
    #     nms = 0.5,
    #     detections_per_img = 100,
    #     box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0)),
    #     cls_agnostic_bbox_reg = False,
    #     bbox_aug_enabled = False
    # )
    postprocessor = make_retinanet_postprocessor(cfg, BoxCoder(weights=(10.0, 10.0, 5.0, 5.0)), False)

    temporal_ens_list = glob.glob(str_folder+'*')
    #-----------------filter train db
    json_file = './MS_COCO/annotations/instances_valminusminival2014.json'
    with open(json_file,'r') as f:
        json_info = json.load(f)
    img_name_list = [_img['file_name'].replace('.jpg','') for _img in json_info['images']]
    temporal_ens = []
    for _path in temporal_ens_list:
        _img_id = os.path.basename(_path)
        if _img_id in img_name_list:
            temporal_ens.append(_path)
    print('initial end ----------------------')
    #----------------------------end
    predcit_dict = {}
    # single 
    for _iter in tqdm(temporal_ens):
        # if _iter.find('/003636')<0:
        #     continue
        pts = glob.glob(os.path.join(_iter,'*.pt'))
        img_id = os.path.basename(_iter)
        pts_iter = [int(_id.split('_x')[-1].replace('.pt','')) for _id in pts]
        idx_sorted = np.argsort(pts_iter)
        pts_sorted = np.array(pts)[idx_sorted]

        #bbox = one_pt_scores(pts_sorted,postprocessor)
        #bbox = multi_pt_scores(pts_sorted,postprocessor)
        #bbox = checks(pts_sorted,postprocessor)
        bbox = multi_align_ens(pts_sorted,postprocessor)

        # try:
        #     bbox = multi_align_ens(pts_sorted,postprocessor)
        # except:
        #     print('except in file',_iter)

        predcit_dict[img_id] = bbox[0].to('cpu')

    #parallel 
    # pool = mp.Pool(mp.cpu_count())
    # predcit_list = [pool.apply(multi_process, args=(_iter, postprocessor)) for _iter in tqdm(temporal_ens)]
    # for _item in predcit_list:
    #     predcit_dict[_item[0]] = _item[1][0]

    torch.save(predcit_dict,'tmp.pt')
    predcit_dict = torch.load('tmp.pt')
    mAp_scores = mAP(cfg,predcit_dict)
    print('mAp is ',mAp_scores)
    print('process_end')



if __name__ == "__main__":
    main()
