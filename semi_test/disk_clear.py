import glob
import os
import numpy as np
import shutil
import torch
from tqdm import tqdm

keep_last = 5
filter_int = 390000
filter_int2 = 180000

str_ens_folder = glob.glob('./model_path/*')
pt_count = []
for i,str_folder in enumerate(str_ens_folder):
    # if i < 105834:
    #     continue
    print(i)
    pts = glob.glob(os.path.join(str_folder,'*.pt'))
    pt_count.append(len(pts))
    #[torch.load(_pt) for _pt in pts]
    if len(pts) == 0:
        continue
    if len(pts) < keep_last:
        continue

    # pts_iter = [int(_iter.split('_x')[-1].replace('.pt','')) for _iter in pts]
    # idx_sorted = np.argsort(pts_iter)
    # pts_sorted = np.array(pts)[idx_sorted][:len(pts) - keep_last]
    # [os.remove(_file) for _file in pts_sorted]
    #-----------------filter by range
    pts_iter = [int(_iter.split('_x')[-1].replace('.pt','')) for _iter in pts]
    sel_ind = np.asarray(pts_iter)>filter_int
    
    pts_sel = np.asarray(pts)[sel_ind]
    [os.remove(_file) for _file in pts_sel]

print(np.amax(pt_count),np.amin(pt_count))
