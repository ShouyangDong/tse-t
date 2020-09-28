import torch
import numpy as np
import shutil
import os
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
import matplotlib.pyplot as plt



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



def main():
    result_file = 'model_path.pth'
    input_img_folder = './input_image_folder'
    output_folder = './folder'
    config_file = './retinanet_R-50-FPN_1x_coco_unlabeled.yaml'
    result_predict = torch.load(result_file)
    jpg_output = './output_jpg'

    cfg.merge_from_file(config_file)
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)[0]

    score_thr = 0.43
    secore_sel = 0.3
    Zeros_obj = 0
    hist_means = []
    hist_ious = []
    Scores_minor = 0
    ious_hard = 0

    sel_file_id = []
    for _id,_bbox in enumerate(result_predict):
        print(_id)
        img_info = data_loaders_val.dataset.get_img_info(_id)
        sel_num = (_bbox.get_field('scores') > secore_sel).sum()

        img_src = os.path.join(output_folder,img_info['file_name'])
        # if img_src.find('000000335584')<0:
        #     continue

        if sel_num < 1:
            Zeros_obj += 1
            # if not os.path.exists(img_src):
            #     continue
            #shutil.copy(img_src,jpg_output)
            continue

        #caculate means
        sel_scores = _bbox.get_field('scores')[_bbox.get_field('scores') > secore_sel]
        mean_scores = sel_scores.mean().numpy()*100
        hist_means.append(mean_scores)
        if mean_scores < (score_thr*100):
            Scores_minor += 1
            # if not os.path.exists(img_src):
            #     continue
            # shutil.copy(img_src,jpg_output)
            continue

        #caculate ious 
        ind_sel = _bbox.get_field('scores') > secore_sel
        box_sel = BoxList(_bbox.bbox[ind_sel],_bbox.size)
        ious = boxlist_iou(box_sel,box_sel) - torch.eye(len(box_sel))
        ious_scores = ious.mean()*1000
        hist_ious.append(ious_scores)
        if ious_scores > 150:
            ious_hard += 1
            # if not os.path.exists(img_src):
            #     continue
            # shutil.copy(img_src,jpg_output)
            continue
        sel_file_id.append(img_info['file_name'])

    
    plt.hist(hist_ious, bins=5)
    plt.gca().set(title='Frequency Histogram of Diamond Depths', ylabel='Frequency')
    plt.savefig('./test2.jpg')
    np.save('sel_unlabeled_ids_r101.npy',sel_file_id)

    print('zeros object = ',Zeros_obj,'Scores_minor',Scores_minor,'ious_hard',ious_hard,'total sample',len(result_predict),'select sample',len(sel_file_id))






if __name__ == "__main__":
    main()