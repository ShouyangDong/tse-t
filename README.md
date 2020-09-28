# Temporal Self-Ensembling Teacher for Semi-Supervised Object Detection


This repository contains the code  for [Temporal Self-Ensembling Teacher for Semi-Supervised Object Detection](https://arxiv.org/abs/2007.06144), by Cong Chen, Shouyang Dong, Ye Tian, Kunlin Cao, Li Liu, Yuanhao Guo, arXiv arXiv:2007.06144:

If you use the code in this repository for a published research project, please cite this paper.

The code is designed to run on Pytorch and Python using the dependencies listed in requirements.txt. You can install the dependencies by running pip install -r requirements.txt
For the relevent packages about this detection framework, please find installation instructions for this repository in INSTALL.md.

## Introducetion
 We propose a novel method Temporal Self-Ensembling Teacher (TSE-T) for SSOD. Differently from previous KD based methods, we devise a temporally evolved teacher model. First, our teacher model ensembles its temporal predictions for unlabeled images under stochastic perturbations. Second, our teacher model ensembles its temporal model weights with the student model weights by an exponential moving average (EMA) which allows the teacher gradually learn from the student. These self-ensembling strategies increase data and model diversity, thus improving teacher predictions on unlabeled images. Finally, we use focal loss to formulate consistency regularization term to handle the data imbalance problem, which is a more efficient manner to utilize the useful information from unlabeled images than a simple hard-thresholding method which solely preserves confident predictions.

## Reference 
The retina network is reference by https://github.com/facebookresearch/Detectron
