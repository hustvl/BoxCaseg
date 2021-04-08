# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Modified by Bin Hu (binhu_19@hust.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import get_world_size, get_rank

from tqdm import tqdm
from core.DUT_eval.quan_eval_demo import dut_eval
from core.pd_loss import PairwiseLoss, mil_loss, psd_loss
from PIL import Image
from skimage import io
import json
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

import base64
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import pandas as pd

pairwise_loss_processor = PairwiseLoss()
bce_loss = nn.BCEWithLogitsLoss()


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


# ----------------------- combine pascal and DUT-TR dataset train and validation ------------------------------------
def train_pd(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
             trainloader, optimizer, model, writer_dict, device, only_weak=False, step_iter=1):
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    model.zero_grad()
    if rank == 0:
        print('Optimizer steps Every {} iterations!!'.format(step_iter))
    for i_iter, batch in enumerate(trainloader):
        images, labels, weakly_flags = batch['img'], batch['label'], batch['weakly']
        if only_weak:
            weakly_flags = torch.ones(images.shape[0])

        f_index = (weakly_flags == 0)  # [0, 1, ... , 0, 1] uint8
        w_index = (weakly_flags != 0)

        f_flag = (torch.sum(f_index).item() > 0)

        images = images.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        images = images.to(device)
        labels = labels.to(device)
        h, w = labels.shape[-2:]

        output, output_f = model(images, labels, f_index)

        # weakly-supervised loss
        output_upsample = F.interpolate(input=output, size=(h, w), mode='bilinear')
        loss_mil = mil_loss(output_upsample, labels)
        loss_pairwise = pairwise_loss_processor(output_upsample)

        # salient full-supervised loss
        output_f_upsample = F.interpolate(input=output_f, size=(h, w), mode='bilinear')
        if f_flag:
            loss_f = bce_loss(output_f_upsample, labels[f_index])
        else:
            # probs_f only contain img[0] output
            loss_f = 0.0 * bce_loss(output_f_upsample, labels[0].unsqueeze(0))  # avoid bp error

        # total loss
        losses = loss_mil + loss_pairwise + loss_f
        loss = losses.mean()

        reduced_loss = reduce_tensor(loss)

        loss.backward()
        if (i_iter + 1) % step_iter == 0:
            optimizer.step()
            model.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}'.format(
                epoch, num_epoch, i_iter, epoch_iters,
                batch_time.average(), lr, print_loss)
            logging.info(msg)

            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate_pd_new(config, testloader, model, writer_dict, device, gt_mask_root=None, mode='evaluate', only_weak=False):
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    prediction_dir = './pascal_prediction_{}/'.format(config.DATASET.DATASET)
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    print('test_prediction_dir:', prediction_dir)

    result_ls = []
    with torch.no_grad():
        for i_test, batch in enumerate(tqdm(testloader)):
            # sample = {'cls': bbox_cls, 'bbox': bbox, 'img_name': image_name,
            #           'img': crop_img, 'ori_img_shape': ori_img_shape}

            image = batch['img']
            image = image.type(torch.FloatTensor)
            image = image.to(device)

            weakly_flags = torch.zeros(batch['img'].shape[0])
            f_index = (weakly_flags == 0)

            pred_w, pred_f = model(image, image[:, 0, :, :].unsqueeze(1), f_index)
            if only_weak:
                pred = pred_w[:, 0, :, :].sigmoid()
            else:
                # pred = (0.0 * pred_w[:, 0, :, :] + 1.0 * pred_f[:, 0, :, :]).sigmoid()
                pred = pred_f[:, 0, :, :].sigmoid()
            pred = normPRED_new(pred)

            batch_result_ls = deal_output(pred, batch)
            result_ls.extend(batch_result_ls)
            # if i_test==20:
            #     break

        print(len(result_ls))
        out_path = prediction_dir + 'seg_result_{}.json'.format(mode)
        with open(out_path, 'w') as fo:
            json.dump(result_ls, fo)
        

        if mode == 'evaluate':
            logging.info('Successfully prediction! Save pseudo mask file of validation images in {}'.format(out_path))
            miou_per_class, miou, iou25, iou50, iou70, iou75 = cal_cls_iou_new(result_ls,
                                                                               gt_mask_root)
            # logging.info('miou_per_class:', miou_per_class)
            logging.info('--------------------------------------------------')
            logging.info('{:<8}{:>5}'.format('miou*:', round(miou, 3)))
            logging.info('{:<8}{:>5}'.format('iou25:', round(iou25, 3)))
            logging.info('{:<8}{:>5}'.format('iou50:', round(iou50, 3)))
            logging.info('{:<8}{:>5}'.format('iou70:', round(iou70, 3)))
            logging.info('{:<8}{:>5}'.format('iou75:', round(iou75, 3)))
            logging.info('--------------------------------------------------')
        else:
            logging.info('Successfully prediction! Save pseudo mask file of training images  in {}'.format(out_path))


def validate_coco(config, testloader, model, writer_dict, device, mode='evaluate', only_weak=False):
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    prediction_dir = './coco_prediction_{}/'.format(config.DATASET.DATASET)
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    print('test_prediction_dir:', prediction_dir)

    result_ls = []
    with torch.no_grad():
        for i_test, batch in enumerate(tqdm(testloader)):
            # sample = {'cls': bbox_cls, 'bbox': bbox, 'img_name': image_name,
            #           'img': crop_img, 'ori_img_shape': ori_img_shape}

            image = batch['img']
            image = image.type(torch.FloatTensor)
            image = image.to(device)

            weakly_flags = torch.zeros(batch['img'].shape[0])
            f_index = (weakly_flags == 0)

            pred_w, pred_f = model(image, image[:, 0, :, :].unsqueeze(1), f_index)
            if only_weak:
                pred = pred_w[:, 0, :, :].sigmoid()
            else:
                pred = pred_f[:, 0, :, :].sigmoid()
            pred = normPRED_new(pred)

            batch_result_ls = deal_output(pred, batch)
            result_ls.extend(batch_result_ls)
            # if i_test==20:
            #     break

        print(len(result_ls), 'instances segmentation results!')
        out_path = prediction_dir + 'seg_result_{}.json'.format(mode)
        with open(out_path, 'w') as fo:
            json.dump(result_ls, fo)

        if mode == 'evaluate':
            print('Successfully prediction! Save pseudo mask file of validation images in {}'.format(out_path))
            miou_per_class, miou, iou25, iou50, iou70, iou75 = cal_cls_iou_coco(result_ls, testloader)

            # print('miou_per_class:', miou_per_class)
            logging.info('--------------------------------------------------')
            logging.info('{:<8}{:>5}'.format('miou*:', round(miou, 3)))
            logging.info('{:<8}{:>5}'.format('iou25:', round(iou25, 3)))
            logging.info('{:<8}{:>5}'.format('iou50:', round(iou50, 3)))
            logging.info('{:<8}{:>5}'.format('iou70:', round(iou70, 3)))
            logging.info('{:<8}{:>5}'.format('iou75:', round(iou75, 3)))
            logging.info('--------------------------------------------------')
        else:
            logging.info('Successfully prediction! Save pseudo mask file of training images  in {}'.format(out_path))


# --------------------------------------- END ------------------------------------------------------------
def normPRED_new(d):
    ma, _ = torch.max(d.view(d.shape[0], -1), dim=1)
    mi, _ = torch.min(d.view(d.shape[0], -1), dim=1)
    ma = ma.view(ma.shape[0], 1, 1)
    mi = mi.view(mi.shape[0], 1, 1)
    dn = (d - mi) / (ma - mi)

    return dn


def save_pascal_output(i_test, pred, data_test, prediction_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    bbox_tensor = data_test['bbox']  # xyxy
    bbox = [i.item() for i in bbox_tensor]
    h, w = data_test['ori_img_shape']
    img_name = data_test['img_name']
    cls = data_test['cls']

    im = Image.fromarray(predict_np * 255).convert('L')
    imo = im.resize((bbox[2] - bbox[0], bbox[3] - bbox[1]), resample=Image.BILINEAR)  # im.resize((width, hight))
    imo = np.array(imo) / 255. > 0.5

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np.array(imo).astype(np.uint8)

    json_dict = {'img_name': img_name[0], 'bbox': bbox, 'cls': cls.item(), 'mask': mask.tolist()}

    json_file = os.path.join(prediction_dir, str(i_test) + '.json')
    json_data = json.dumps(json_dict)
    with open(json_file, 'w') as fo:
        fo.write(json_data)


def deal_output(pred, data_test):
    # sample = {'cls': bbox_cls, 'bbox': bbox, 'img_name': image_name,
    #           'img': crop_img, 'ori_img_shape': ori_img_shape}
    predict_np = pred.cpu().data.numpy()

    result_ls = []
    for n in range(pred.shape[0]):
        bbox_tensor = data_test['bbox']  # xyxy list(xmin_tensor, xmax_tensor, ymin_tensor, ymax_tensor)
        bbox = [i[n].item() for i in bbox_tensor]
        # bbox_l_tensor = data_test['bbox_l']  # xyxy list(xmin_tensor, xmax_tensor, ymin_tensor, ymax_tensor)
        # bbox_l = [i[n].item() for i in bbox_l_tensor]
        # import pdb; pdb.set_trace()
        h, w = data_test['ori_img_shape'][0][n].item(), data_test['ori_img_shape'][1][n].item()
        img_name = data_test['img_name'][n]
        cls = data_test['cls'][n]

        im = Image.fromarray(predict_np[n] * 255).convert('L')
        imo = im.resize((bbox[2] - bbox[0], bbox[3] - bbox[1]), resample=Image.BILINEAR)  # im.resize((width, hight))
        imo = np.array(imo) / 255. > 0.5

        mask_tmp = np.zeros((h, w), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np.array(imo).astype(np.uint8)
        # mask_tmp[bbox_l[1]:bbox_l[3], bbox_l[0]:bbox_l[2]] = imo.astype(np.uint8)
        # mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask_tmp[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        rle = mask_util.encode(np.array(mask[:, :, np.newaxis], order="F"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")

        result_ls.extend([
            {'img_name': img_name,
             'cls': cls.item(),
             'bbox': bbox,
             'segmentation': rle,
             'img_shape': (h, w)}
        ])

    return result_ls


def cal_mask_iou(mask, gt_mask):
    """
    mask(np.array):(w*h)
    gt_mask(np.array):(w*h)
    """
    inters = np.sum(mask * gt_mask)
    uni = np.sum(mask) + np.sum(gt_mask) - inters
    iou = inters.astype(np.float) / (uni.astype(np.float) + 1e-7)
    return iou


def cal_cls_iou_new(result_ls, gt_mask_root=None):
    if gt_mask_root is None:
        gt_mask_root = '/home/xinggang/hb/WSIS_BBTP_get_psd/Dataset/VOCSBD/VOC2012/SegmentationObject'
    # pred_json_root = os.path.join('/home/xinggang/hb/U-2-Net/pascal_u2net_val/test_data/', json_root_name)
    obj_num = 0
    iou25_num = 0
    iou50_num = 0
    iou70_num = 0
    iou75_num = 0

    last_img_name = None
    img_bbox_num = 0
    iou_ls = [[] for i in range(20)]
    for i in tqdm(range(len(result_ls))):
        pred_dict = result_ls[i]
        cls = pred_dict['cls']  # 1-20
        mask = annToMask(pred_dict, pred_dict['img_shape'][0], pred_dict['img_shape'][1])
        img_name = pred_dict['img_name']
        img_gt_mask_name = img_name.replace('jpg', 'png')
        if last_img_name is None:
            last_img_name = img_name
        elif last_img_name == img_name:
            img_bbox_num += 1
        else:
            img_bbox_num = 0
            last_img_name = img_name

        # get gt mask
        gt_mask_path = os.path.join(gt_mask_root, img_gt_mask_name)
        gt_mask = (np.array(Image.open(gt_mask_path)) == (img_bbox_num + 1))

        iou = cal_mask_iou(mask, gt_mask)
        iou_ls[cls - 1].append(iou)
        obj_num += 1
        iou25_num = iou25_num + 1 if iou >= 0.25 else iou25_num
        iou50_num = iou50_num + 1 if iou >= 0.50 else iou50_num
        iou70_num = iou70_num + 1 if iou >= 0.70 else iou70_num
        iou75_num = iou75_num + 1 if iou >= 0.75 else iou75_num

    miou_per_class = [sum(iou_ls_per_cls) / len(iou_ls_per_cls) if len(iou_ls_per_cls) != 0 else 0 for iou_ls_per_cls in
                      iou_ls]
    miou = sum(miou_per_class) / len(miou_per_class)
    iou25 = iou25_num / float(obj_num)
    iou50 = iou50_num / float(obj_num)
    iou70 = iou70_num / float(obj_num)
    iou75 = iou75_num / float(obj_num)

    return miou_per_class, miou, iou25, iou50, iou70, iou75


def cal_cls_iou_coco(result_ls, testloader):
    gt_dict = testloader.dataset.coco_json_dict['annotations']
    obj_num = 0
    visual_num = 0
    iou25_num = 0
    iou50_num = 0
    iou70_num = 0
    iou75_num = 0

    iou_ls = [[] for i in range(90)]
    box_iou_ls = [[] for i in range(90)]

    for i in tqdm(range(len(result_ls))):
        pred_dict = result_ls[i]
        cls = pred_dict['cls']
        if 'box_iou' in pred_dict.keys():
            box_iou = pred_dict['box_iou']
            box_iou_ls[cls - 1].append(box_iou)
        mask = annToMask(pred_dict, pred_dict['img_shape'][0], pred_dict['img_shape'][1])
        # get gt mask
        gt_mask = annToMask(gt_dict[i], pred_dict['img_shape'][0], pred_dict['img_shape'][1])
        iou = cal_mask_iou(mask, gt_mask)
        iou_ls[cls - 1].append(iou)
        obj_num += 1
        iou25_num = iou25_num + 1 if iou >= 0.25 else iou25_num
        iou50_num = iou50_num + 1 if iou >= 0.50 else iou50_num
        iou70_num = iou70_num + 1 if iou >= 0.70 else iou70_num
        iou75_num = iou75_num + 1 if iou >= 0.75 else iou75_num

    miou_per_class = [sum(iou_ls_per_cls) / len(iou_ls_per_cls) if len(iou_ls_per_cls) != 0 else 0 for iou_ls_per_cls in
                      iou_ls]
    miou = sum(miou_per_class) / 80  # len(miou_per_class)
    iou25 = iou25_num / float(obj_num)
    iou50 = iou50_num / float(obj_num)
    iou70 = iou70_num / float(obj_num)
    iou75 = iou75_num / float(obj_num)

    return miou_per_class, miou, iou25, iou50, iou70, iou75


def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = mask_util.decode(rle)
    return m


def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_util.frPyObjects(segm, height, width)
        rle = mask_util.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = mask_util.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle


if __name__ == '__main__':
    pass
