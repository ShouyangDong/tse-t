# Temporal Self-Ensembling Teacher for Semi-Supervised Object Detection


This repository contains the code  for [Temporal Self-Ensembling Teacher for Semi-Supervised Object Detection](https://arxiv.org/abs/2007.06144), Shouyang Dong, by Cong Chen, Ye Tian, Kunlin Cao, Li Liu, Yuanhao Guo, arXiv arXiv:2007.06144:

If you use the code in this repository for a published research project, please cite this paper.

The code is designed to run on Pytorch and Python using the dependencies listed in requirements.txt. You can install the dependencies by running pip install -r requirements.txt
For the relevent packages about this detection framework, please find installation instructions for this repository in INSTALL.md.

## Introducetion
we propose the temporal model ensemble based semi-supervised object detection. In order to obtain a feasible semi-supervised object detection model, we adapt the standard teacher-student architecture. The teacher model guides the student model producing similar predictions on unlabeled data. We propose the temporal model ensemble (tse-t) module to construct the teacher model. 

## Reference 
The retina network is reference by https://github.com/facebookresearch/Detectron
