import os

import torch
import torch.utils.data
from PIL import Image
import sys
import glob
import numpy as np
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
#nms
from maskrcnn_benchmark.modeling.roi_heads.box_head.inference import PostProcessor
from semi_test.post_process import multi_pt_scores,ens_teacher
import multiprocessing as mp
mp.set_start_method('spawn', True)


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def load_temporal_files(img_id,folder,ens_num=5):
    str_file = os.path.join(folder,img_id)
    pts = glob.glob(os.path.join(str_file,'*.pt'))
    if(len(pts)) < 1:
        return []

    pts_iter = np.asarray([int(_iter.split('_x')[-1].replace('.pt','')) for _iter in pts])
    #pts_iter[pts_iter > 180000] = 0
    idx_sorted = np.argsort(pts_iter)
    pts_sorted_all = np.array(pts)[idx_sorted] 
    pts_sorted = pts_sorted_all[-ens_num:]
    #pts_rm = pts_sorted_all[:-ens_num]
    #_ = [os.remove(_pt) for _pt in pts_rm]


    bbox_ts = []
    for  _pt in pts_sorted:
        try:
            _bb = torch.load(_pt,map_location=torch.device("cpu"))
        except Exception as e:
            print('error in file ',_pt)
            raise e
        bbox_ts.append(_bb)
    return bbox_ts

def ensemble_bboxes(bbox_ts,im_sz,anchor_strides,sel_threshold,device):

    if len(bbox_ts)<1:
        img_tmp = BoxList(torch.zeros([0,4]),[400,400])
        img_tmp.add_field('sel_ind',[])
        return [img_tmp,img_tmp,img_tmp,img_tmp,img_tmp]

    bbox_ts = [[_result_layer.to_half().to(device) for _result_layer in  _result_img] for _result_img in  bbox_ts]

    ens_dict = ens_teacher(bbox_ts,im_sz,anchor_strides,sel_threshold=sel_threshold)
    bboxes = []
    for ens_box,feat_w_h,ens_score,sel_ind in zip(ens_dict['ens_boxes'],ens_dict['feat_w_h'],ens_dict['ens_scores'],ens_dict['sel_ind']):
        if(len(sel_ind)<1):
            ens_box = torch.empty([0,4]).to('cpu')
        bbox = BoxList(ens_box.to('cpu'),ens_dict['image_size'])
        bbox.add_field('feat_w_h',feat_w_h.to('cpu'))
        bbox.add_field('ens_scores',ens_score.to('cpu'))
        bbox.add_field('sel_ind',sel_ind.to('cpu'))
        bboxes.append(bbox)
    bboxes = [_result_layer.to_float32() for _result_layer in  bboxes]
    return bboxes 

def load_temporal_ens_id(img_id,folder,sel_threshold,ens_num=5):
    device='cuda'
    str_file = os.path.join(folder,img_id)
    pts = glob.glob(os.path.join(str_file,'*.pt'))
    if(len(pts)) < 1:
        img_tmp = BoxList(torch.zeros([0,4]),[400,400])
        img_tmp.add_field('sel_ind',[])
        return [img_tmp,img_tmp,img_tmp,img_tmp,img_tmp]

    pts_iter = np.asarray([int(_iter.split('_x')[-1].replace('.pt','')) for _iter in pts])
    idx_sorted = np.argsort(pts_iter)
    pts_sorted_all = np.array(pts)[idx_sorted] 
    pts_sorted = pts_sorted_all[-ens_num:]
    #pts_rm = pts_sorted_all[:-ens_num]
    #_ = [os.remove(_pt) for _pt in pts_rm]

    #bbox = one_pt_scores(pts_sorted,postprocessor)
    # pool = mp.Pool(ens_num)
    # bbox_ts = [pool.apply(torch.load, args=(_iter,)) for _iter in pts_sorted]

    bbox_ts = []
    for  _pt in pts_sorted:
        try:
            _bb = torch.load(_pt)
        except Exception as e:
            print('error in file ',_pt)
            raise e
        bbox_ts.append(_bb)
    #bbox_ts = [[__bbox.to(device) for __bbox in _boxes] for _boxes in bbox_ts]
    _ = [[_result_layer.to_float32() for _result_layer in  _result_img] for _result_img in  bbox_ts]

    ens_dict = ens_teacher(bbox_ts,sel_threshold=sel_threshold)
    bboxes = []
    for ens_box,feat_w_h,ens_score,sel_ind in zip(ens_dict['ens_boxes'],ens_dict['feat_w_h'],ens_dict['ens_scores'],ens_dict['sel_ind']):
        if(len(sel_ind)<1):
            ens_box = torch.empty([0,4]).to('cpu')
        bbox = BoxList(ens_box.to('cpu'),ens_dict['image_size'])
        bbox.add_field('feat_w_h',feat_w_h.to('cpu'))
        bbox.add_field('ens_scores',ens_score.to('cpu'))
        bbox.add_field('sel_ind',sel_ind.to('cpu'))
        bboxes.append(bbox)
    #_ = [_result_layer.to_float32() for _result_layer in  bboxes]
    return bboxes

class PascalVOCDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self, data_dir, split,temporal_saved_path, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))
        self.temporal_saved_path = temporal_saved_path
        self.postprocessor = PostProcessor(
            score_thresh = 0.05,
            nms = 0.5,
            detections_per_img = 100,
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0)),
            cls_agnostic_bbox_reg = False,
            bbox_aug_enabled = False
        )

        self.__getitem__(0)
        self.get_img_info(0)

    def __getitem__(self, index):
        #index = 12#np.random.randint(100)
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")
        img_sz = img.size

        if (self.root.replace('datasets/voc/','') + '_' + self.image_set).find('VOC2012_test')<0:
            target = self.get_groundtruth(index)
            target = target.clip_to_image(remove_empty=True)
        else:
            target = BoxList(torch.zeros([1,4],dtype=torch.float32), img_sz, mode="xyxy")  

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        bboxes = load_temporal_ens_id(img_id,self.temporal_saved_path,sel_threshold=0.3,ens_num=5)
        return img, target,[self.root.replace('datasets/voc/','') + '_' + self.image_set ,bboxes,index]
        
    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]

        if (self.root.replace('datasets/voc/','') + '_' + self.image_set).find('VOC2012_test')>-1:
            img = Image.open(self._imgpath % img_id).convert("RGB")
            return {"height": img.size[1], "width": img.size[0]}

        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return PascalVOCDataset.CLASSES[class_id]
