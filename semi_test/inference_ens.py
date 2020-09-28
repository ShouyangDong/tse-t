# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size
from maskrcnn_benchmark.utils.comm import all_gather
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.engine.bbox_aug import im_detect_bbox_aug
from maskrcnn_benchmark.data.datasets.concat_dataset import ConcatDataset
from maskrcnn_benchmark.data.datasets.voc import ensemble_bboxes

from semi_test.post_process import concat_layers_nms
from maskrcnn_benchmark.data.transforms import trans_reverse



id_to_txt = {1: 'person',10: 'traffic light',11: 'fire hydrant',13: 'stop sign',14: 'parking meter',15: 'bench',16: 'bird',17: 'cat',18: 'dog',19: 'horse',2: 'bicycle',\
            20: 'sheep',21: 'cow',22: 'elephant',23: 'bear',24: 'zebra',25: 'giraffe',27: 'backpack',28: 'umbrella',3: 'car',31: 'handbag',32: 'tie',33: 'suitcase',34: 'frisbee',\
            35: 'skis',36: 'snowboard',37: 'sports ball',38: 'kite',39: 'baseball bat',4: 'motorcycle',40: 'baseball glove',41: 'skateboard',42: 'surfboard',43: 'tennis racket',\
            44: 'bottle',46: 'wine glass',47: 'cup',48: 'fork',49: 'knife',5: 'airplane',50: 'spoon',51: 'bowl',52: 'banana',53: 'apple',54: 'sandwich',55: 'orange',56: 'broccoli',\
            57: 'carrot',58: 'hot dog',59: 'pizza',6: 'bus',60: 'donut',61: 'cake',62: 'chair',63: 'couch',64: 'potted plant',65: 'bed',67: 'dining table',7: 'train',70: 'toilet',\
            72: 'tv',73: 'laptop',74: 'mouse',75: 'remote',76: 'keyboard',77: 'cell phone',78: 'microwave',79: 'oven',8: 'truck',80: 'toaster',81: 'sink',82: 'refrigerator',84: 'book',\
            85: 'clock',86: 'vase',87: 'scissors',88: 'teddy bear',89: 'hair drier',9: 'boat',90: 'toothbrush',}

contiguous_category_id_to_json_id = {1: 1,10: 10,11: 11,12: 13,13: 14,14: 15,15: 16,16: 17,17: 18,18: 19,19: 20,2: 2,20: 21,21: 22,22: 23,23: 24,24: 25,25: 27,26: 28,27: 31,28: 32,
                                    29: 33,3: 3,30: 34,31: 35,32: 36,33: 37,34: 38,35: 39,36: 40,37: 41,38: 42,39: 43,4: 4,40: 44,41: 46,42: 47,43: 48,44: 49,45: 50,46: 51,47: 52,
                                    48: 53,49: 54,5: 5,50: 55,51: 56,52: 57,53: 58,54: 59,55: 60,56: 61,57: 62,58: 63,59: 64,6: 6,60: 65,61: 67,62: 70,63: 72,64: 73,65: 74,66: 75,
                                    67: 76,68: 77,69: 78,7: 7,70: 79,71: 80,72: 81,73: 82,74: 84,75: 85,76: 86,77: 87,78: 88,79: 89,8: 8,80: 90,9: 9,}



def map_to_img_coco(data_loader,idx_info):
    # db_id_img_id = [data_loader.dataset.get_idxs(_id) for _id in idx]
    # idx_name = [data_loader.dataset.datasets[db_id].id_to_img_map[img_id] for db_id,img_id in db_id_img_id]
    # db_idx = np.array(db_id_img_id)[:,0]
    # img_idx = np.array(db_id_img_id)[:,1]
    img_idx = [_id[-1] for _id in idx_info]
    id_map = {'coco/train2014':0, 'coco/val2014':0,'coco/unlabeled2017':0}
    db_idx = [id_map[_name[0]] for _name in idx_info]
    if isinstance(data_loader.dataset,ConcatDataset):
        datamap = data_loader.dataset.datasets
    else:
        datamap = [data_loader.dataset]
    img_name = [datamap[db_id].get_img_info(img_id)['file_name'].replace('.jpg','') for db_id,img_id in zip(db_idx,img_idx)]

    temporal_ens_bboxes = [_id[1] for _id in idx_info]
    return np.array(db_idx),np.array(img_idx),img_name,temporal_ens_bboxes

def write_img(str_jpg,predict_bboxes,vis_thr=0.3):
    jpg_output = 'output_image_folder'
    input_img_folder = 'input_image_folder'
    img_c = cv2.imread(os.path.join(input_img_folder,str_jpg+'.jpg'))

    json_id = [contiguous_category_id_to_json_id[_id] for _id in predict_bboxes.get_field('labels').numpy()]
    txt_id = [id_to_txt[_id] for _id in json_id]

    for _bbox,_score,_txt in zip(predict_bboxes.bbox,predict_bboxes.get_field('scores').numpy(),txt_id):
        if _score < vis_thr:
            continue
        img_c = cv2.rectangle(img_c, (_bbox[0],_bbox[1]), (_bbox[2],_bbox[3]), (0, 255, 0), 1)
        img_c = cv2.putText(img_c, str(round(_score, 2))+'_'+_txt, (_bbox[0],_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 1)
    
    cv2.imwrite(os.path.join(jpg_output,str_jpg+'.jpg'), img_c)    




def compute_on_dataset(model, data_loader,postprocessor, semi_loss,anchor_strides,device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    print('--------------------------------total batch ',len(data_loader))
    for iteration, (images, targets_with_trans_info, idx) in enumerate(tqdm(data_loader)):
        # if iteration < 8000:
        #     continue

        db_idx,img_idx,idx_name,bboxes_batch = map_to_img_coco(data_loader,idx)

        try:
            temporal_ens_bboxes = [ensemble_bboxes(_boxes,_im_sz,anchor_strides,0.05,device) for _boxes,_im_sz in zip(bboxes_batch,images.image_sizes)]

        except Exception as e:
            print('error in file ',idx_name,img_idx)
            raise e
        # if iteration != 149:
        #     continue
        #----------get target 
        targets = [_iter[0] for _iter in targets_with_trans_info]
        trans_param = [_iter[1] for _iter in targets_with_trans_info]
        target_origin = [trans_reverse(_res,_info).to('cpu') for _res,_info in zip(targets,trans_param)]
        #------------


        bbox_ts_gpu = [[__bbox.to(device) for __bbox in _boxes] for _boxes in temporal_ens_bboxes]
        output_ens = concat_layers_nms(temporal_ens_bboxes,postprocessor)
        output_ens = [o.to(cpu_device) for o in output_ens]

        #_ = [write_img(str_jpg,predict_bboxes) for str_jpg,predict_bboxes in  zip(idx_name,output_ens)]

        # with torch.no_grad():
        #     output = model(images.to(device))
        # output = [trans_reverse(_res,_info).to(cpu_device) for _res,_info in zip(output,trans_param)]
        # targets = [trans_reverse(_res,_info).to(cpu_device) for _res,_info in zip(targets,trans_param)]

        # # _ = [ _output.add_field('class_logits',(F.one_hot(_output.get_field('labels'),80).float().T*_output.get_field('scores')).T) for _output in output]
        # # _ = [ _output.add_field('class_logits',(F.one_hot(_output.get_field('labels'),80).float().T*_output.get_field('scores')).T) for _output in output_ens]
        
        # loss = semi_loss(output,targets)
        # loss_ens = semi_loss(output_ens,targets)

        # out_str = ''

        # for _idx,_loss in zip(img_idx,loss):
        #     out_str = out_str + 'img_idx = {}, loss = {};'.format(_idx,_loss)
        # print('model predict',out_str)
        # out_str = ''
        # for _idx,_loss in zip(img_idx,loss_ens):
        #     out_str = out_str + 'img_idx = {}, loss = {};'.format(_idx,_loss)
        # print('ensem predict',out_str)

        # for _key,_box in zip(idx_name,output):
        #     if len(_box) > 0 and (not all(_box.size)):
        #         print(_key)
                
        results_dict.update(
            {img_id: result for img_id, result in zip(img_idx, output_ens)}
        )
        #break

    #torch.save(results_dict,'tmp_pk/results_dict.pt')
    #results_dict = torch.load('tmp_pk/results_dict.pt')

    # for _key in results_dict.keys():
    #     if len(results_dict[_key]) > 0 and (not all(results_dict[_key].size)):
    #         print(_key)
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    # if len(image_ids) != image_ids[-1] + 1:
    #     logger = logging.getLogger("maskrcnn_benchmark.inference")
    #     logger.warning(
    #         "Number of images that were gathered from multiple processes is not "
    #         "a contiguous set. Some images might be missing from the evaluation"
    #     )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        postprocessor,
        semi_loss,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        anchor_strides=None
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader,postprocessor,semi_loss,anchor_strides, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
