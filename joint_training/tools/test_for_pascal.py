# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil

import time
import timeit

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import validate_pd_new
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, get_rank

# import glob
from torchvision import transforms
from hybrid_data_loader import PascalDataset, ResizeT_T, ToTensor_T


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
    parser.add_argument('--voc_val_json_file',
                        help="path of voc val json file",
                        required=True,
                        type=str)
    parser.add_argument('--voc_image_folder',
                        help="path of voc train image folder",
                        required=True,
                        type=str)
    parser.add_argument('--voc_val_mask_root',
                        help="path of voc val mask root",
                        required=True,
                        type=str)
    parser.add_argument('--fcn_ratio',
                        help="alpha in paper",
                        default=0.7,
                        type=float)
    parser.add_argument('--model_pth',
                        help="the path of model file",
                        required=True,
                        type=str)
    parser.add_argument('--mode',
                        help="\'eval\' for evaluating and generating pseudo mask, otherwise generate pseudo mask file only.",
                        required=True,
                        type=str)

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

    # prepare data
    json_root = args.voc_val_json_file  # '/home/xinggang/hb/WSIS_BBTP_get_psd/Dataset/VOCSBD/voc_2012_val_cocostyle.json'
    image_set_root = args.voc_image_folder  # "/home/xinggang/hb/WSIS_BBTP_get_psd/Dataset/VOCSBD/VOC2012/JPEGImages"
    gt_mask_root = args.voc_val_mask_root  # '/home/xinggang/hb/WSIS_BBTP_get_psd/Dataset/VOCSBD/VOC2012/SegmentationObject'
    test_dataset = PascalDataset(json_root,
                                 image_set_root,
                                 transform=transforms.Compose([ResizeT_T(320),
                                                               ToTensor_T()])
                                 )

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=None)

    model = FullModel(model, None)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    model_state_file = args.model_pth
    if os.path.isfile(model_state_file):
        logger.info("=> loaded trained model ({})"
                    .format(model_state_file))
        checkpoint = torch.load(model_state_file,
                                map_location=lambda storage, loc: storage)
        model.module.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint (epoch {})"
                    .format(checkpoint['epoch']))

    # start = timeit.default_timer()
    start_time = time.time()
    validate_pd_new(config, testloader, model, writer_dict, device,
                    gt_mask_root, mode=args.mode, only_weak=False)

    print('used_time:', int(time.time() - start_time))


if __name__ == '__main__':
    main()
