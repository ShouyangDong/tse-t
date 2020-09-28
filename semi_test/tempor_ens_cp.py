import glob
import numpy as np
from tqdm import tqdm
import os
import shutil

src_folder = './tempor_pred_save/'
target_folder = './tempor_pred_save_0_17500/'
iter_start = 17500
iter_stop = 25000
if not os.path.exists(target_folder):
    os.mkdir(target_folder)

temporal_ens = glob.glob(src_folder+'*')

for _iter in tqdm(temporal_ens):
    pts = glob.glob(os.path.join(_iter,'*.pt'))
    if len(pts)<1:
        continue

    img_id = os.path.basename(_iter)
    pts_iter = [int(_iter.split('_x')[-1].replace('.pt','')) for _iter in pts]
    sel_ind = np.logical_and(np.array(pts_iter) > iter_start , np.array(pts_iter) < iter_stop)
    cp_files = np.array(pts)[sel_ind]

    target_img_folder = os.path.join(target_folder,img_id)
    
    if not os.path.exists(target_img_folder):
        os.mkdir(target_img_folder)

    for _im in cp_files:
        _im_id = os.path.basename(_im)
        _img_target = os.path.join(target_img_folder,_im_id)
        shutil.copy(_im,_img_target)


