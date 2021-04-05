# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import train_pd
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, get_rank

import glob
from hybrid_data_loader import CDTrainDataset, PDtransform
from torchvision import transforms
from combined_random_sampler import DistributedCombinedRandomSampler, CombinedRandomSampler


# ----------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1234)
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--only_weak',
                        help="only use weakly branch",
                        default=False,
                        type=bool)
    parser.add_argument('--weakly_ratio',
                        help="the ratio of weakly data in a batch",
                        type=float)
    parser.add_argument('--coco_train_root',
                        help="path of COCO train image folder",
                        required=True,
                        type=str)
    parser.add_argument('--coco_train_json',
                        help="path of COCO train json file",
                        required=True,
                        type=str)
    parser.add_argument('--coco_overlap_file',
                        help="path of COCO overlap file",
                        required=True,
                        type=str)
    parser.add_argument('--dut_image_folder',
                        help="path of dut train image folder",
                        required=True,
                        type=str)
    parser.add_argument('--dut_label_folder',
                        help="path of dut train label folder",
                        required=True,
                        type=str)
    parser.add_argument('--fcn_ratio',
                        help="alpha in paper",
                        required=True,
                        type=float)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    device = torch.device('cuda:{}'.format(args.local_rank))

    # build model
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config, fcn_ratio=args.fcn_ratio)

    if args.local_rank == 0:
        # provide the summary of model
        dump_input = torch.rand(
            (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
        )
        logger.info(get_model_summary(model.to(device), dump_input.to(device)))

        # copy model file
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    if True:  # distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    # prepare data
    # COCO2017 train
    coco_train_root = args.coco_train_root  # "/data/xinggang/coco/coco/train2017/"
    coco_train_json = args.coco_train_json  # "/data/xinggang/coco/coco/annotations/instances_train2017.json"
    overlap_file = args.coco_overlap_file  # "/home/xinggang/hb/HRNet-Semantic-Segmentation/tools/overlap_anno_id.json"

    # DUTS-TR
    dut_image_dir = args.dut_image_folder  # '/home/xinggang/hb/U-2-Net/train_data/data/DUTS-TR/DUTS-TR-Image-single/'
    dut_label_dir = args.dut_label_folder  # '/home/xinggang/hb/U-2-Net/train_data/data/DUTS-TR/DUTS-TR-Mask-single/'

    image_ext = '.jpg'
    label_ext = '.png'

    tra_img_name_list = glob.glob(dut_image_dir + '*' + image_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split("/")[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(dut_label_dir + imidx + label_ext)

    dut_img_name_list = tra_img_name_list
    dut_lbl_name_list = tra_lbl_name_list

    cd_train_dataset = CDTrainDataset(coco_train_root=coco_train_root,
                                    coco_train_json=coco_train_json,
                                    img_name_list=dut_img_name_list,
                                    lbl_name_list=dut_lbl_name_list,
                                    transform=transforms.Compose([PDtransform(320, 288)])
                                    )

    num_w, num_f = cd_train_dataset.get_num_wf()

    if args.local_rank == 0:
        print("---")
        print("The number of all train images: ", len(cd_train_dataset))
        print("The number of weakly-supervised train images: ", num_w)
        print("The number of salient train images: ", num_f)
        print("---")

    if distributed:
        # train_sampler = DistributedSampler(cd_train_dataset)
        train_sampler = DistributedCombinedRandomSampler(num_w, num_f, config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus), args.weakly_ratio, overlap_file=overlap_file)
    else:
        # train_sampler = None
        train_sampler = CombinedRandomSampler(num_w, num_f, config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus), args.weakly_ratio, overlap_file=overlap_file)

    trainloader = torch.utils.data.DataLoader(
        cd_train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    # full model
    model = FullModel(model, None)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD([{'params':
                                          filter(lambda p: p.requires_grad,
                                                 model.parameters()),
                                      'lr': config.TRAIN.LR}],
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int32(num_f / (1-args.weakly_ratio)/config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            # best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        if distributed:
            train_sampler.set_epoch(epoch)

        train_pd(config, epoch, config.TRAIN.END_EPOCH,
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict,
                  device, only_weak=args.only_weak)

        if (epoch + 1) % 5 == 0: # and args.local_rank == 0:
            if args.local_rank == 0:
                logger.info('=> saving val checkpoint to {}'.format(
                    final_output_dir + 'checkpoint_epoch_{}.pth.tar'.format(epoch+1)))
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(final_output_dir, 'checkpoint_epoch_{}.pth.tar'.format(epoch+1)))

        if args.local_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

            if epoch == end_epoch - 1:
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'final_state.pth'))

                writer_dict['writer'].close()
                end = timeit.default_timer()
                logger.info('Hours: %d' % np.int32((end - start) / 3600))
                logger.info('Done')


if __name__ == '__main__':
    main()
