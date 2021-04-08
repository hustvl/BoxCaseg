# pd_loss
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

import time

import pdb
from tqdm import tqdm
import os
from skimage import io, transform
from PIL import Image
# from miou_cal import cal_cls_iou
import json
import logging

bce_loss = nn.BCEWithLogitsLoss()


# weakly supervise
def mil_loss(d, label_v):
    """
    :param d: [N, 1, h, w]  logits
    :param label_v: [N, 1, h, w]
    :return:
    """
    h, w = d.shape[-2:]
    d_row_max = F.max_pool2d(d, (h, 1)).view((d.shape[0], d.shape[1], -1))
    d_col_max = F.max_pool2d(d, (1, w)).view((d.shape[0], d.shape[1], -1))
    d_max = torch.cat((d_row_max, d_col_max), dim=2)
    label_row_max = F.max_pool2d(label_v, (h, 1)).view((d.shape[0], d.shape[1], -1))
    label_col_max = F.max_pool2d(label_v, (1, w)).view((d.shape[0], d.shape[1], -1))
    label_bag = torch.cat((label_row_max, label_col_max), dim=2)
    return bce_loss(d_max, label_bag)


class PairwiseLoss(object):
    def __init__(self):
        self.center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.],
                                           [0., 0., 0.]])  # , device=device)

        # TODO: modified this as one conv with 8 channels for efficiency
        self.pairwise_weights_list = [
            torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [0., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 1.],
                          [0., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 1., 0.], [0., 0., 0.],
                          [0., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 0.],
                          [0., 1., 0.]]),  # , device=device),
            torch.tensor([[1., 0., 0.], [0., 0., 0.],
                          [0., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 0., 1.], [0., 0., 0.],
                          [0., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 0.],
                          [1., 0., 0.]]),  # , device=device),
            torch.tensor([[0., 0., 0.], [0., 0., 0.],
                          [0., 0., 1.]]),  # , device=device),
        ]

    def __call__(self, d):
        device = d.device
        d = d.sigmoid()
        pairwise_loss = []
        for w in self.pairwise_weights_list:
            conv = torch.nn.Conv2d(1, 1, 3, bias=False, padding=(1, 1))
            weights = self.center_weight - w
            weights = weights.view(1, 1, 3, 3).to(device)
            conv.weight = torch.nn.Parameter(weights)
            for param in conv.parameters():
                param.requires_grad = False
            aff_map = conv(d)

            cur_loss = (aff_map ** 2)
            cur_loss = torch.mean(cur_loss)
            pairwise_loss.append(cur_loss)
        pairwise_loss = torch.mean(torch.stack(tuple(pairwise_loss)))
        return 0.05 * pairwise_loss


def psd_loss(weak_d_prob, full_d_prob, fg_thres=0.85, bg_thres=0.05):
    """

    :param weak_d_prob: [N, 1, H, W] weakly branch probs output about weakly supervised dataset
    :param full_d_prob: [N, 1, H, W] full branch probs output about weakly supervised dataset
    :param fg_thres: threshold of foreground
    :param bg_thres: threshold of background
    :return:
    """
    full_d_prob = full_d_prob.clamp(1e-4, 1 - 1e-4)
    weak_d_prob = normPRED(weak_d_prob)
    fg_count = torch.sum((weak_d_prob > fg_thres)).item()
    bg_count = torch.sum((weak_d_prob < bg_thres)).item()
    fg_loss = -torch.sum((weak_d_prob > fg_thres).to(torch.float)*torch.log(full_d_prob))
    bg_loss = -torch.sum((weak_d_prob < bg_thres).to(torch.float)*torch.log(1-full_d_prob))
    loss = fg_loss/(fg_count + 1e-4) + bg_loss/(bg_count + 1e-4)
    return loss


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn