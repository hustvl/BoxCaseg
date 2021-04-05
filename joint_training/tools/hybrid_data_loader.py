import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import json
import os
import pdb
import random


# ==========================Train dataset transform==========================
class PDtransform(object):
    def __init__(self, mid_size=320, output_size=288):
        assert isinstance(mid_size, (int, tuple))
        assert isinstance(output_size, (int, tuple))
        self.resize = ResizeT(mid_size)
        self.randomcrop = RandomCrop(output_size)
        self.resize_out = ResizeT(output_size)
        self.totensor = ToTensor()

    def __call__(self, sample):
        weakly_flag = sample['weakly']
        if weakly_flag:
            return self.totensor(self.resize_out(sample))
        else:
            return self.totensor((self.randomcrop(self.resize(sample))))


class ResizeT(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # sample = {'weakly': weakly_flag, 'img_name': image_name, 'img': image, 'label': label}
        weakly_flag, image_name, image, label = sample['weakly'], sample['img_name'], sample['img'], sample['label']
        img = transform.resize(image, (self.output_size[0], self.output_size[1]), mode='constant')
        lbl = transform.resize(label, (self.output_size[0], self.output_size[1]), mode='constant', order=0,
                               preserve_range=True)
        return {'weakly': weakly_flag, 'img_name': image_name, 'img': img, 'label': lbl}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        weakly_flag, image_name, image, label = sample['weakly'], sample['img_name'], sample['img'], sample['label']
        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'weakly': weakly_flag, 'img_name': image_name, 'img': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        weakly_flag, image_name, image, label = sample['weakly'], sample['img_name'], sample['img'], sample['label']
        # pascal data(weakly)
        if weakly_flag:
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            tmpLbl = np.zeros(label.shape)

            if np.max(label) < 1e-6:
                label = label
            else:
                label = label / np.max(label)

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = image[:, :, 0] - 102.9801 / 255.
                tmpImg[:, :, 1] = image[:, :, 1] - 115.9465 / 255.
                tmpImg[:, :, 2] = image[:, :, 2] - 122.7717 / 255.

            tmpLbl[:, :, 0] = label[:, :, 0]
            # change the r,g,b to b,r,g from [0,255] to [0,1]
            # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
            tmpImg = tmpImg.transpose((2, 0, 1))
            tmpLbl = label.transpose((2, 0, 1))

            return {'weakly': weakly_flag, 'img_name': image_name,
                    'img': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

        # DUTS-TR
        else:
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            tmpLbl = np.zeros(label.shape)

            image = image / np.max(image)
            if np.max(label) < 1e-6:
                label = label
            else:
                label = label / np.max(label)

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

            tmpLbl[:, :, 0] = label[:, :, 0]

            # change the r,g,b to b,r,g from [0,255] to [0,1]
            # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
            tmpImg = tmpImg.transpose((2, 0, 1))
            tmpLbl = label.transpose((2, 0, 1))

            return {'weakly': weakly_flag, 'img_name': image_name,
                    'img': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


# ==========================Test dataset transform==========================
class ResizeT_T(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['img']
        img = transform.resize(image, (self.output_size[0], self.output_size[1]), mode='constant')

        sample.update({'img': img})
        return sample


class ToTensor_T(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['img']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = image[:, :, 0] - 102.9801 / 255.
            tmpImg[:, :, 1] = image[:, :, 1] - 115.9465 / 255.
            tmpImg[:, :, 2] = image[:, :, 2] - 122.7717 / 255.

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))

        sample.update({'img': torch.from_numpy(tmpImg)})

        return sample


# ======================  Train dataset  ===========================
class PDTrainDataset(Dataset):
    def __init__(self, json_root, image_set_root, img_name_list, lbl_name_list, transform=None):
        super(PDTrainDataset, self).__init__()

        # pascal_VOCSBD
        self.image_set_root = image_set_root
        with open(json_root, 'r') as fo:
            self.json_dict = json.load(fo)
        self.num_weak = len(self.json_dict['annotations'])

        # DUTS-TR
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.num_full = len(img_name_list)

        self.transform = transform

    def __len__(self):
        # return len(self.json_dict['annotations']) + len(self.image_name_list)
        return self.num_weak + self.num_full

    def get_num_wf(self):
        return self.num_weak, self.num_full

    def __getitem__(self, idx):
        # pascal data
        if idx < self.num_weak:
            bbox_info = self.json_dict['annotations'][idx]
            # get_type
            weakly_flag = True

            # # get class
            # bbox_cls = bbox_info['category_id']

            # get image_name
            image_idx = bbox_info['image_id'] - 1
            image_name = self.json_dict['images'][image_idx]['file_name']

            # get_image
            image_path = os.path.join(self.image_set_root, image_name)
            image = io.imread(image_path)

            # ori_image_shape (h, w)
            ori_img_shape = image.shape[:2]

            # get bbox
            bbox_xywh = bbox_info['bbox']  # xywh

            # convert to bbox xyxy
            bbox = bbox_xywh.copy()
            bbox[2] = bbox_xywh[0] + bbox_xywh[2]
            bbox[3] = bbox_xywh[1] + bbox_xywh[3]
            # bbox_l = bbox_broaden(bbox, ori_img_shape, self.ratio)

            # box shake
            bbox_l = self.bbox_broaden(bbox, ori_img_shape)

            # get crop_img
            crop_img = image[bbox_l[1]: bbox_l[3], bbox_l[0]: bbox_l[2], :]

            # get label
            label = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
            label[bbox[1]: bbox[3], bbox[0]: bbox[2], :] = 255
            label = label[bbox_l[1]: bbox_l[3], bbox_l[0]: bbox_l[2], :]

            sample = {'weakly': weakly_flag, 'img_name': image_path, 'img': crop_img, 'label': label}
        # DUTS-TR
        else:
            image = io.imread(self.image_name_list[idx - self.num_weak])
            image_name = self.image_name_list[idx - self.num_weak]
            weakly_flag = False

            if len(self.label_name_list) == 0:
                label_3 = np.zeros(image.shape)
            else:
                label_3 = io.imread(self.label_name_list[idx - self.num_weak])

            label = np.zeros(label_3.shape[0:2])
            if 3 == len(label_3.shape):
                label = label_3[:, :, 0]
            elif 2 == len(label_3.shape):
                label = label_3

            if 3 == len(image.shape) and 2 == len(label.shape):
                label = label[:, :, np.newaxis]
            elif 2 == len(image.shape) and 2 == len(label.shape):
                image = image[:, :, np.newaxis]
                label = label[:, :, np.newaxis]

            sample = {'weakly': weakly_flag, 'img_name': image_name, 'img': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
            # try:
            #     sample = self.transform(sample)
            # except Exception as e:
            #     print(idx, weakly_flag, bbox, bbox_l, image.shape, image_name)
            #     print(e)
            #     return

        return sample

    def bbox_broaden(self, bbox, ori_img_shape):
        # xyxy
        # import pdb;pdb.set_trace()
        bbox_center_x = (bbox[0] + bbox[2]) / 2.
        bbox_center_y = (bbox[1] + bbox[3]) / 2.

        bbox_half_w = (bbox[2] - bbox[0]) / 2.
        bbox_half_h = (bbox[3] - bbox[1]) / 2.

        bbox_center_x = bbox_center_x + random.uniform(-0.25, 0.25) * bbox_half_w
        bbox_center_y = bbox_center_y + random.uniform(-0.25, 0.25) * bbox_half_h

        ratio_x = random.uniform(0.5, 1.5)
        ratio_y = random.uniform(0.5, 1.5)

        xmin = int(max(0, bbox_center_x - ratio_x * bbox_half_w))
        xmax = int(min(ori_img_shape[1] - 1, bbox_center_x + ratio_x * bbox_half_w))
        ymin = int(max(0, bbox_center_y - ratio_y * bbox_half_h))
        ymax = int(min(ori_img_shape[0] - 1, bbox_center_y + ratio_y * bbox_half_h))
        # print([xmin, ymin, xmax, ymax])
        return [xmin, ymin, xmax, ymax]


class CDTrainDataset(Dataset):
    def __init__(self, coco_train_root, coco_train_json, img_name_list, lbl_name_list, dut_box_path=None, ratio=1.0,
                 transform=None, dut_shake=False):
        super(CDTrainDataset, self).__init__()

        # coco_set
        with open(coco_train_json, 'r') as fo:
            self.coco_train_json = json.load(fo)
        self.coco_train_root = coco_train_root
        self.coco_train_num = len(self.coco_train_json['annotations'])

        # DUTS-TR
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.dut_shake = dut_shake
        self.dut_box_path = dut_box_path
        if dut_box_path:
            with open(dut_box_path, 'r') as fo:
                self.dut_box_ls = json.load(fo)
        else:
            self.dut_box_ls = None

        self.transform = transform

    def get_num_wf(self):
        return self.coco_train_num, len(self.image_name_list)

    def __len__(self):
        return self.coco_train_num + len(self.image_name_list)

    def __getitem__(self, idx):
        # coco data
        if idx < self.coco_train_num:
            ann_dict = self.coco_train_json['annotations'][idx]
            weakly_flag = True
            # get image_name
            image_id = ann_dict['image_id']
            image_name = ('000000000000' + str(image_id))[-12:] + '.jpg'
            # get_image
            image_path = os.path.join(self.coco_train_root, image_name)
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            # ori_image_shape (h, w)
            ori_img_shape = image.shape[:2]
            # get bbox
            bbox_xywh = ann_dict['bbox']  # xywh
            bbox_xywh = [int(i) for i in bbox_xywh]
            # avoid h==0 or w==0
            bbox_xywh[2] = max(bbox_xywh[2], 1)
            bbox_xywh[3] = max(bbox_xywh[3], 1)
            # convert to bbox xyxy
            bbox = bbox_xywh.copy()
            bbox[2] = bbox_xywh[0] + bbox_xywh[2]
            bbox[3] = bbox_xywh[1] + bbox_xywh[3]
            # bbox_l = bbox_broaden(bbox, ori_img_shape, self.ratio)
            bbox_l = self.bbox_broaden(bbox, ori_img_shape)
            # get crop_img
            if len(image.shape) == 3:
                crop_img = image[bbox_l[1]: bbox_l[3], bbox_l[0]: bbox_l[2], :]
            else:
                print(image.shape, image_name)
                crop_img = image[bbox_l[1]: bbox_l[3], bbox_l[0]: bbox_l[2], :]
            # get label
            label = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
            label[bbox[1]: bbox[3], bbox[0]: bbox[2], :] = 255
            label = label[bbox_l[1]: bbox_l[3], bbox_l[0]: bbox_l[2], :]

            sample = {'weakly': weakly_flag, 'img_name': image_path, 'img': crop_img, 'label': label}

        # DUTS-TR
        else:
            image = io.imread(self.image_name_list[idx - self.coco_train_num])
            image_name = self.image_name_list[idx - self.coco_train_num]
            weakly_flag = False

            if (0 == len(self.label_name_list)):
                label_3 = np.zeros(image.shape)
            else:
                label_3 = io.imread(self.label_name_list[idx - self.coco_train_num])

            label = np.zeros(label_3.shape[0:2])
            if (3 == len(label_3.shape)):
                label = label_3[:, :, 0]
            elif (2 == len(label_3.shape)):
                label = label_3

            if (3 == len(image.shape) and 2 == len(label.shape)):
                label = label[:, :, np.newaxis]
            elif (2 == len(image.shape) and 2 == len(label.shape)):
                image = image[:, :, np.newaxis]
                label = label[:, :, np.newaxis]

            if self.dut_shake and self.dut_box_ls:
                last_name = image_name.split('/')[-1]
                bbox = self.dut_box_ls[last_name]
                bbox_l = self.bbox_broaden(bbox, image.shape[:2])
                # print(idx-25815, bbox, bbox_l, image.shape[:2])
                # get crop_img
                crop_image = image[bbox_l[1]: bbox_l[3], bbox_l[0]: bbox_l[2], :]
                # get label
                label = label[bbox_l[1]: bbox_l[3], bbox_l[0]: bbox_l[2], :]
                # print(image.shape[:2], bbox_l)
                sample = {'weakly': weakly_flag, 'img_name': image_name, 'img': crop_image, 'label': label}
            else:
                sample = {'weakly': weakly_flag, 'img_name': image_name, 'img': image, 'label': label}

        if self.transform:
            try:
                sample = self.transform(sample)
            except Exception as e:
                print(idx, bbox, bbox_l, image.shape, image_name)
                print(e)
                return

        return sample

    def bbox_broaden(self, bbox, ori_img_shape):
        # xyxy
        # import pdb;pdb.set_trace()
        bbox_center_x = (bbox[0] + bbox[2]) / 2.
        bbox_center_y = (bbox[1] + bbox[3]) / 2.

        bbox_half_w = (bbox[2] - bbox[0]) / 2.
        bbox_half_h = (bbox[3] - bbox[1]) / 2.

        bbox_center_x = bbox_center_x + random.uniform(-0.25, 0.25) * bbox_half_w
        bbox_center_y = bbox_center_y + random.uniform(-0.25, 0.25) * bbox_half_h

        ratio_x = random.uniform(0.5, 1.5)
        ratio_y = random.uniform(0.5, 1.5)

        xmin = int(max(0, bbox_center_x - ratio_x * bbox_half_w))
        xmax = int(min(ori_img_shape[1] - 1, bbox_center_x + ratio_x * bbox_half_w))
        ymin = int(max(0, bbox_center_y - ratio_y * bbox_half_h))
        ymax = int(min(ori_img_shape[0] - 1, bbox_center_y + ratio_y * bbox_half_h))

        if xmax == xmin:
            if xmin == 0:
                xmax = 1
            else:
                xmin -= 1

        if ymax == ymin:
            if ymin == 0:
                ymax = 1
            else:
                ymin -= 1
        # print([xmin, ymin, xmax, ymax])
        return [xmin, ymin, xmax, ymax]


# ======================  Test dataset  ===========================
class PascalDataset(Dataset):
    def __init__(self, json_root, image_set_root, transform=None):
        super(PascalDataset, self).__init__()

        self.image_set_root = image_set_root
        with open(json_root, 'r') as fo:
            self.json_dict = json.load(fo)
        self.transform = transform

    def __len__(self):
        return len(self.json_dict['annotations'])

    def __getitem__(self, bbox_idx):
        bbox_info = self.json_dict['annotations'][bbox_idx]
        # get class
        bbox_cls = bbox_info['category_id']
        # get image_name
        image_idx = bbox_info['image_id'] - 1
        image_name = self.json_dict['images'][image_idx]['file_name']
        # get_image
        image_path = os.path.join(self.image_set_root, image_name)
        image = io.imread(image_path)
        # ori_image_shape (h, w)
        ori_img_shape = image.shape[:2]
        # get bbox
        bbox_xywh = bbox_info['bbox']  # xywh
        # convert to bbox xyxy
        bbox = bbox_xywh.copy()
        bbox[2] = bbox_xywh[0] + bbox_xywh[2]
        bbox[3] = bbox_xywh[1] + bbox_xywh[3]

        # get crop_img
        crop_img = image[bbox[1]: bbox[3], bbox[0]: bbox[2], :]

        sample = {'cls': bbox_cls, 'bbox': bbox, 'img_name': image_name,
                  'img': crop_img, 'ori_img_shape': ori_img_shape}

        if self.transform:
            sample = self.transform(sample)

        return sample


class COCODataset(Dataset):
    def __init__(self, coco_json, coco_image_set_root, transform=None):
        super(COCODataset, self).__init__()

        # coco
        self.coco_image_set_root = coco_image_set_root
        with open(coco_json, 'r') as fo:
            self.coco_json_dict = json.load(fo)

        self.transform = transform

    def __len__(self):
        return len(self.coco_json_dict['annotations'])

    def __getitem__(self, idx):
        ann_dict = self.coco_json_dict['annotations'][idx]
        # get image_name
        image_id = ann_dict['image_id']
        image_name = ('000000000000' + str(image_id))[-12:] + '.jpg'
        # get_image
        image_path = os.path.join(self.coco_image_set_root, image_name)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        # ori_image_shape (h, w)
        ori_img_shape = image.shape[:2]
        # get cls
        bbox_cls = ann_dict['category_id']
        # get bbox
        bbox_xywh = ann_dict['bbox']  # xywh
        bbox_xywh = [int(i) for i in bbox_xywh]
        # avoid h==0 or w==0
        bbox_xywh[2] = max(bbox_xywh[2], 1)
        bbox_xywh[3] = max(bbox_xywh[3], 1)
        # convert to bbox xyxy
        bbox = bbox_xywh.copy()
        bbox[2] = bbox_xywh[0] + bbox_xywh[2]
        bbox[3] = bbox_xywh[1] + bbox_xywh[3]
        # get crop_img
        if len(image.shape) == 3:
            crop_img = image[bbox[1]: bbox[3], bbox[0]: bbox[2], :]
        else:
            print(image.shape, image_name)
            crop_img = image[bbox[1]: bbox[3], bbox[0]: bbox[2], :]

        sample = {'cls': bbox_cls, 'bbox': bbox, 'img_name': image_name,
                  'img': crop_img, 'ori_img_shape': ori_img_shape}
        if self.transform:
            try:
                sample = self.transform(sample)
            except:
                print(sample['img'].shape, sample['img_name'], bbox_xywh, idx)

        return sample


if __name__ == '__main__':
    pass
