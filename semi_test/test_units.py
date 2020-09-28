# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from semi_test.inference_ens import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.modeling.rpn.retinanet.inference import make_retinanet_postprocessor
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.data.datasets import UnlabeledDataset
from semi_test.semi_loss import make_semi_box_loss_evaluator
# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')
import multiprocessing
multiprocessing.set_start_method('spawn', True)


if __name__ == "__main__":
    ds = UnlabeledDataset('/MS_COCO/annotations/image_info_unlabeled2017.json','datasets/coco/unlabeled2017','/tempor_pred_coco_unlabeled/')