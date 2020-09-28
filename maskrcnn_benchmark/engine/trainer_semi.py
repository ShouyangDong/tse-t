# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os
import numpy as np
import torch
import torch.distributed as dist
import copy
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.data.transforms import trans_reverse
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.data.datasets.concat_dataset import ConcatDataset
from maskrcnn_benchmark.data.datasets.voc import ensemble_bboxes

from semi_test.post_process import check_resuts
from semi_test.semi_loss import semi_loss_fn

from apex import amp

DATASET = 'coco'
N_CLS = 80

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def predict_collect_postprocess(post_processor, proposals,trans_param):
    result = [post_processor((_proposal.get_field('class_logits'), _proposal.get_field('box_regression')), [_proposal]) for _proposal in proposals]
    result_origin = [trans_reverse(_res[0],_info).to('cpu') for _res,_info in zip(result,trans_param)]
    return result_origin
# def decode(rpn_cls,box_encode,anchors,box_coder):
#     #inputs:
#     #rpn_cls: feat num * (batch,4*class_num,feat_hight,feat_wight)
#     #box_encode: feat num * (batch,4*anchor_nu,feat_hight,feat_wight)
#     #anchors: batch* (feat num*(4, anchor_nu*feat_hight*feat_wight) )
#     #box_coder: function 
#     #returns
#     #rpn_cls, batch* (feat num*(num_cls,anchor_nu,feat_hight,feat_wight) ) 
#     #rpn_bbox, batch* (feat num*(4,anchor_nu,feat_hight,feat_wight))
#     result = []
#     for _b in range(len(anchors)): 
#         result_single = []
#         for _l in range(len(anchors[_b])):
#             bbox_decode = box_coder(box_encode[_l][_b].view([-1,4]),anchors[_b][_l].bbox)
#             result_single.append({'bbox':bbox_decode, 'rpn_cls':rpn_cls })
#         result.append(result_single)
#     return result_single



def predict_retina_postprocess(post_process_mod,box_coder,result_dict,trans_param,image_size):

    boxes = post_process_mod(result_dict['anchors'],result_dict['rpn_cls'],result_dict['rpn_box_encode'],image_size)
    
    result_origin = [[trans_reverse(_res_ij,_info).to('cpu') for _res_ij in _res]  for _res,_info in zip(boxes,trans_param)]
    #aa = [[check_resuts(_ll.to('cpu')) for _ll in _l] for _l in boxes]
    _ = [[_result_layer.to_half() for _result_layer in  _result_img] for _result_img in  result_origin]
    return result_origin



def update_ema_variables(model, ema_model, alpha, global_step):
    world_size = get_world_size() 
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    iCount = 0
    for ema_param, param in zip(ema_model.parameters(recurse=True), model.parameters(recurse=True)):
        if world_size > 1:
            update_param = param.data.clone()
            torch.distributed.all_reduce(update_param, op=torch.distributed.ReduceOp.SUM)
            update_param /= world_size
        else:
            update_param = param.data
        ema_param.data.mul_(alpha).add_(1 - alpha, update_param)



def map_to_img_voc(data_loader,idx_info):
    # db_id_img_id = [data_loader.dataset.get_idxs(_id) for _id in idx]
    # idx_name = [data_loader.dataset.datasets[db_id].id_to_img_map[img_id] for db_id,img_id in db_id_img_id]
    # db_idx = np.array(db_id_img_id)[:,0]
    # img_idx = np.array(db_id_img_id)[:,1]
    img_idx = [_id[-1] for _id in idx_info]
    id_map = {"VOC2007_train":0, "VOC2007_val":1,"VOC2012_train":2,"VOC2012_val":3,"VOC2012_test":4}
    db_idx = [id_map[_name[0]] for _name in idx_info]
    if isinstance(data_loader.dataset,ConcatDataset):
        datamap = data_loader.dataset.datasets
    else:
        datamap = [data_loader.dataset]
    img_name = [datamap[db_id].id_to_img_map[img_id] for db_id,img_id in zip(db_idx,img_idx)]

    temporal_ens_bboxes = [_id[1] for _id in idx_info]
    return np.array(db_idx),np.array(img_idx),img_name,temporal_ens_bboxes

def map_to_img_coco(data_loader,idx_info):
    # db_id_img_id = [data_loader.dataset.get_idxs(_id) for _id in idx]
    # idx_name = [data_loader.dataset.datasets[db_id].id_to_img_map[img_id] for db_id,img_id in db_id_img_id]
    # db_idx = np.array(db_id_img_id)[:,0]
    # img_idx = np.array(db_id_img_id)[:,1]
    img_idx = [_id[-1] for _id in idx_info]
    id_map = {'coco/train2014':0, 'coco/val2014':1}
    db_idx = [id_map[_name[0]] for _name in idx_info]
    if isinstance(data_loader.dataset,ConcatDataset):
        datamap = data_loader.dataset.datasets
    else:
        datamap = [data_loader.dataset]
    img_name = [datamap[db_id].get_img_info(img_id)['file_name'].replace('.jpg','') for db_id,img_id in zip(db_idx,img_idx)]

    temporal_ens_bboxes = [_id[1] for _id in idx_info]
    return np.array(db_idx),np.array(img_idx),img_name,temporal_ens_bboxes

def map_to_img_coco_unlabeled(data_loader,idx_info):
    # db_id_img_id = [data_loader.dataset.get_idxs(_id) for _id in idx]
    # idx_name = [data_loader.dataset.datasets[db_id].id_to_img_map[img_id] for db_id,img_id in db_id_img_id]
    # db_idx = np.array(db_id_img_id)[:,0]
    # img_idx = np.array(db_id_img_id)[:,1]
    img_idx = [_id[-1] for _id in idx_info]
    id_map = {'coco/train2014':0, 'coco/val2014':1,'coco/unlabeled2017':2}
    db_idx = [id_map[_name[0]] for _name in idx_info]
    if isinstance(data_loader.dataset,ConcatDataset):
        datamap = data_loader.dataset.datasets
    else:
        datamap = [data_loader.dataset]
    img_name = [datamap[db_id].get_img_info(img_id)['file_name'].replace('.jpg','') for db_id,img_id in zip(db_idx,img_idx)]

    temporal_ens_bboxes = [_id[1] for _id in idx_info]
    return np.array(db_idx),np.array(img_idx),img_name,temporal_ens_bboxes




def map_to_img(data_loader,idx_info):
    if DATASET is 'coco':
        return map_to_img_coco_unlabeled(data_loader,idx_info)#return map_to_img_coco(data_loader,idx_info)
    else:
        return map_to_img_voc(data_loader,idx_info)
    return None

def semi_weight_by_epoch_voc(iteration,start_iter = 60000):
    if iteration < (start_iter + 10000):
        return 0.
    if iteration < (start_iter + 20000):
        return 1
    if iteration < (start_iter + 40000):
        return 2.
    if iteration < (start_iter + 50000):
        return 4.
    if iteration < (start_iter + 60000):
        return 8.

    return 10.





def sigmoid_rampup(current_shift,rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current_shift = np.clip(current_shift, 0.0, rampup_length)
        phase = 1.0 - current_shift / rampup_length
        return float(np.exp(-5 * phase * phase))


def semi_weight_by_epoch(iteration,start_iter = 0,rampup_length = 70000,consistence_weight = 0.4,consistence_trunc=1.):
    if iteration < start_iter:
        return 0.
    # if iteration > 500000:
    #     return 4.

    # if iteration > 465000:
    #     return 4.

    # if iteration > 917500:
    #     return 4.
    # if iteration > 790000:
    #     return 2.

    # if iteration > 350000:
    #     return 1.

    return min(consistence_weight * sigmoid_rampup(iteration - start_iter, rampup_length - start_iter),consistence_trunc)


   

# def semi_weight_by_epoch_step(iteration,start_iter = 0):
#     restart_iter = iteration - start_iter
#     epoch_iter = 27210 #train 82081, val 35185,unlabeled 100428,  / 4gpu*2 batch == 27210
#     # range [0,3],save ensemble result,  domain

#     if restart_iter < (epoch_iter*3): #[0,3]
#         return 0.,epoch_iter*3 + start_iter
#     start_epoch,end_epoch,rampup_epoch,pk_save_epoch = (3,5,)
#     if restart_iter < (epoch_iter*end_epoch):  #[3,5]
#         return semi_weight_by_epoch(restart_iter,start_iter = start_epoch*epoch_iter,rampup_length = 4*epoch_iter,consistence_weight = 0.4),epoch_iter*4+start_iter

#     if restart_iter < (epoch_iter*9):  #[5,9]
#         return semi_weight_by_epoch(restart_iter,start_iter = 5*epoch_iter,rampup_length = 6*epoch_iter,consistence_weight = 1.),epoch_iter*5+start_iter

#     if restart_iter < (epoch_iter*13):  #[9,13]
#         return semi_weight_by_epoch(restart_iter,start_iter = 5*epoch_iter,rampup_length = 6*epoch_iter,consistence_weight = 1.),epoch_iter*5+start_iter

#     return 10.


def do_train(
    model,
    model_ema,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    local_rank,
    checkpoint_period,
    cfg_arg,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    meters_ema = MetricLogger(delimiter="  ")

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    ema_decay = arguments["ema_decay"]
    loss_semi = arguments['loss_semi']
    temporal_save_path = cfg_arg["temporal_save_path"]
    model.train()
    model_ema.train()
    box_coder = BoxCoder(weights=(10., 10., 5., 5.))
    temporal_ens = {}
    start_training_time = time.time()
    end = time.time()
    labeled_database = arguments["HYPER_PARAMETERS"]['LABELED_DATABASE']
    temporal_supervised_losses = []

    for iteration, (images, targets_with_trans_info, idx) in enumerate(data_loader, start_iter):
        targets = [_iter[0] for _iter in targets_with_trans_info]
        trans_info = [_iter[1] for _iter in targets_with_trans_info]
        
        try:
            db_idx,img_idx,idx_name,bboxes_batch = map_to_img(data_loader,idx)
            temporal_ens_bboxes = [ensemble_bboxes(_boxes,_im_sz,arguments["ANCHOR_STRIDES"],arguments["HYPER_PARAMETERS"]['ENS_THRE'],device) for _boxes,_im_sz in zip(bboxes_batch,images.image_sizes)]

            img_size = [(_sz[1],_sz[0]) for _sz in images.image_sizes]
            pred_trans_info = copy.deepcopy(trans_info)
            temporal_ens_pred = []


            for i,_sz in enumerate(img_size):
                pred_trans_info[i][1] = _sz
                temporal_ens_per = [trans_reverse(_temporal_ens,pred_trans_info[i]).to(device) for _temporal_ens in temporal_ens_bboxes[i]]
                temporal_ens_pred.append(temporal_ens_per)

            db_w = []
            for i,_db in enumerate(db_idx):
                if _db not in labeled_database:
                    _bbox = BoxList(torch.zeros([1,4]), (images.image_sizes[i][1],images.image_sizes[i][0]), mode="xyxy")
                    _bbox.add_field('labels',torch.ones([1]))
                    targets[i] = _bbox
                    db_w.append(0.)
                else:
                    db_w.append(1.)


            if any(len(target) < 1 for target in targets):
                logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
                continue
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration


            images = images.to(device)
            targets = [target.to(device) for target in targets]
            update_ema_variables(model, model_ema, ema_decay, iteration)

            _loss_dict,result = model(images, targets)
            #---------------------loss masked by 
            with torch.no_grad():
                _loss_dict_ema,result_ema = model_ema(images, targets)
                is_labeled_db_weight = torch.tensor(db_w,dtype=torch.float32).to(device)

            loss_dict = {}
            loss_dict_ema = {}
            for _key in _loss_dict.keys():
                loss_dict[_key] = torch.sum(torch.stack(_loss_dict[_key],dim=0) * is_labeled_db_weight)
                loss_dict_ema[_key] = torch.sum(torch.stack(_loss_dict_ema[_key],dim=0) * is_labeled_db_weight)

            # loss_dict = _loss_dict
            # loss_dict_ema = _loss_dict_ema

            #result_origin = [trans_reverse(_res,_info) for _res,_info in zip(result_ema,trans_info)]
            #result_origin = predict_collect_postprocess(arguments['postprocess'],result_ema,trans_info)
            result_origin = predict_retina_postprocess(arguments['postprocess'],box_coder,result_ema,trans_info,images.image_sizes)

            # any_zeros = [_iter.bbox.shape[0] == 0 for _iter in temporal_ens_pred]
            # if any(any_zeros):
            #     loss_dict['semi_box_reg'] = torch.tensor(0,dtype=torch.float32,device=device)
            #     loss_dict['semi_cls'] = torch.tensor(0,dtype=torch.float32,device=device)
            # else:
            #     semi_loss = loss_semi(
            #         result, temporal_ens_pred)
            #     for _key in semi_loss.keys():
            #         loss_dict[_key] = torch.sum(torch.stack(semi_loss[_key],dim=0) * (1 - db_weight)) * arguments["semi_weight"]


            #balance losses
            with torch.no_grad():
                supversed_loss = (loss_dict['loss_retina_cls'] + loss_dict['loss_retina_reg'])/(np.sum(db_w) + 0.1)
            temporal_supervised_losses.append(supversed_loss)
            temporal_supervised_losses = temporal_supervised_losses[-100:]
            sup_loss = torch.stack(temporal_supervised_losses).mean()
            meters.update(sup_loss=sup_loss)

            if get_world_size() > 1:
                torch.distributed.all_reduce(torch.stack(temporal_supervised_losses).mean(), op=torch.distributed.ReduceOp.SUM)
            balance_weight = min(1./(sup_loss/0.28)**12,1.)

            semi_loss = semi_loss_fn(
                result,result_ema, temporal_ens_pred,images.image_sizes,box_coder,n_cls=arguments["HYPER_PARAMETERS"]['NCLS'],reg_cons_w=arguments["HYPER_PARAMETERS"]['REG_CONSIST_WEIGHT'])
            semi_loss_weight = semi_weight_by_epoch(iteration,start_iter = arguments["HYPER_PARAMETERS"]['EPOCH_BATCH_NUM']*arguments["HYPER_PARAMETERS"]['START_ITER'],
                                    rampup_length = arguments["HYPER_PARAMETERS"]['EPOCH_BATCH_NUM']*arguments["HYPER_PARAMETERS"]['RAMPUP_LENGTH'],
                                    consistence_weight = arguments["HYPER_PARAMETERS"]['CONSISTENCE_WEIGHT'],
                                    consistence_trunc = arguments["HYPER_PARAMETERS"]['MAX_CONSISTENT_LOSS'])#semi_weight_by_epoch(iteration)
            for _key in semi_loss.keys():
                #loss_dict[_key] = torch.sum(semi_loss[_key] * (1 - is_labeled_db_weight))*semi_loss_weight*balance_weight # not used labeled 
                loss_dict[_key] = torch.sum(semi_loss[_key])*semi_loss_weight

            for i,(_id,_labeled) in enumerate(zip(idx_name,db_w)):
                # if _labeled == 1:
                #     continue
                result_dict = {'iteration':iteration,'result':result_origin[i]}
                if _id in temporal_ens.keys():
                    temporal_ens[_id].append(result_dict)
                else:
                    temporal_ens[_id] = [result_dict]


            #print('id={},{},scores={}----------{}'.format(idx_name[0],idx_name[1],result_origin[0].get_field('objectness')[:5],result_origin[1].get_field('objectness')[:5]))
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)

            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)


            loss_dict_reduced_ema = reduce_loss_dict(loss_dict_ema)
            losses_reduced_ema = sum(loss for loss in loss_dict_reduced_ema.values())
            meters_ema.update(loss=losses_reduced_ema, **loss_dict_reduced_ema)

            optimizer.zero_grad()
            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()

            if not iteration < arguments["HYPER_PARAMETERS"]['EPOCH_BATCH_NUM']*arguments["HYPER_PARAMETERS"]['START_ITER']:
                optimizer.step()
            #scheduler.step()


            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}","{meters_ema}",
                            "lr: {lr:.6f}",
                            "semi_w:{semi_w:2.3f}",
                            "supervised loss{sup_loss:2.3f},"
                            "balance_weight{balance_weight:2.3f},"
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        meters_ema = str(meters_ema),
                        lr=optimizer.param_groups[0]["lr"],
                        semi_w=semi_loss_weight,
                        sup_loss=sup_loss,
                        balance_weight=balance_weight,
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )


            if (iteration - 50) % 100  == 0:
                for _key in temporal_ens.keys():
                    for _iter in temporal_ens[_key]:
                        str_folder = os.path.join(temporal_save_path,_key)#"{}/{}".format(temporal_save_path,_key)
                        str_file = '{}/{}_loc{}_iter_x{:07d}.pt'.format(str_folder,_key,local_rank,_iter['iteration'])
                        if not os.path.exists(str_folder):
                            os.makedirs(str_folder)
                        torch.save(_iter['result'],str_file)
                        del _iter['result']
                        
                del temporal_ens
                temporal_ens = {}

            if iteration % checkpoint_period == 0:
                save_time = time.time()
                checkpointer.save("model_{:07d}".format(iteration), **arguments)

            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)
                
        except Exception as e:
            print('error in file ',idx_name,img_idx)
            raise e


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
