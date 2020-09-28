# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import numpy as np
import os
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from .voc import load_temporal_ens_id,load_temporal_files


class UnlabeledDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root,temporal_saved_path, transforms=None
    ):
        super(UnlabeledDataset, self).__init__(root,ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        filter_ids = np.load(os.path.join(os.path.dirname(ann_file),'sel_unlabeled_ids_r101.npy'))#'sel_unlabeled_ids_v2.npy'
        ids = []
        #save ids ,for load speed
        # for img_id in self.ids:
        #     img_info = self.coco.imgs[img_id]
        #     if any([img_info['width']<400,img_info['height']<400]):
        #         continue
        #     if img_info['file_name'] not in filter_ids:
        #         continue
        #     ids.append(img_id)
        # self.ids = ids
        # np.save(os.path.join(os.path.dirname(ann_file),'sel_unlabeled_ids_r101_map.npy'),ids) #'unlabeled_ids_map_v2.npy'
        #load ids ,for load speed
        ids = np.load(os.path.join(os.path.dirname(ann_file),'sel_unlabeled_ids_r101_map.npy')).tolist()
        self.ids = ids

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.temporal_saved_path = temporal_saved_path
        self.__getitem__(0)


    def __getitem__(self, idx):
        #idx = 8418
        img_infos = self.get_img_info(idx)

        img, anno = super(UnlabeledDataset, self).__getitem__(idx)
        target = BoxList(torch.zeros([1,4]),img.size, mode="xyxy")

        with torch.no_grad():
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            try:
                bboxes = load_temporal_files(img_infos['file_name'].replace('.jpg',''),self.temporal_saved_path,ens_num=5)
            except Exception as e:
                print('error in file ',img_infos['file_name'].replace('.jpg',''))
                raise e

        return img, target, [self.root.replace('datasets/',''),bboxes,idx]

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data




