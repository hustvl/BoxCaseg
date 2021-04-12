# Content

1. [Requirements: hardware](#1)
2. [Basic installation](#2)
3. [Dataset preparation](#3)
4. [Usage](#4)
5. [Our models download](#5)
6. [Some supplements](#6)

## <span id = "1">1. Requirements: hardware</span>

Two Nvidia TITAN V GPUs (12GB RAM），11GB RAM is also OK.

## <span id = "2">2. Basic installation</span>

```
pip install -r requirements.txt
```

**HRNet pretrained model download**
You can download HRNet-W48-C pretrained model from [here](https://github.com/HRNet/HRNet-Image-Classification) and place it in *'$joint_training/pretrained_model/hrnetv2_pretrained.pth'*.

## <span id = "3">3. Dataset preparation</span>

**Augmented Pascal VOC 2012 (10582 images for segmentation)**

This dataset is from [BBTP](https://github.com/chengchunhsu/WSIS_BBTP) and can be downloaded [here](https://drive.google.com/file/d/1lGCVvrst_PVsdG6C57Xz00PF3ge2kJgL/view?usp=sharing).

**DUT-TR-Single (7991images) and DUT-TR-Single-new (4701 images)**

DUT-TR-single-new has no class overlap with Pascal VOC 20 class.

* [BaiDuYun](https://pan.baidu.com/s/1_5i4AAmSKzHSsax9cdDRRw), Access code：oxay
* [GoogleDrive](https://drive.google.com/drive/folders/12qjGTBzTgehf_5GNF5ph0Rdm3o1xfISt?usp=sharing)

**COCO2017**

It's the same as public MS COCO2017 dataset.

Finally， your directory tree should be look like this:

```
./data
├── coco
├── DUTS
│   ├── aug.py
│   ├── check.py
│   ├── DUTS-TR-Image-single
│   ├── DUTS-TR-Image-single-new
│   ├── DUTS-TR-Mask-single
│   └── DUTS-TR-Mask-single-new
└── VOCSBD
      ├── VOC2012
      ├── voc_2012_train_aug_cocostyle.json
      └── voc_2012_val_cocostyle.json
```

## <span id = "4">4. Usage</span>

Run the bash files directly:

### Training

--------

**Jointly training Pascal VOC and DUT-TR-Single with 2 GPUs**

```
bash train_pd.sh
```

**Jointly training Pascal VOC and DUT-TR-Single-new with 2 GPUs**

You need replace follow lines in *$joint_training/train_pd.sh*

```
--dut_image_folder ./data/DUTS/DUTS-TR-Image-single/ \
--dut_label_folder ./data/DUTS/DUTS-TR-Mask-single/ \
```

with

```
--dut_image_folder ./data/DUTS/DUTS-TR-Image-single-new/ \
--dut_label_folder ./data/DUTS/DUTS-TR-Mask-single-new/ \
```

Then it's the same as jointly training Pascal VOC and DUT-TR-Single with 2 GPUs.

```
bash train_pd.sh
```

**Jointly training COCO2017 and DUT-TR-Single with 2 GPUs**

```
bash train_cd.sh
```

### Validation

---

To calculate metric *mIoU^** and *IoU@k* (k ∈ {0.25, 0.50, 0.70, 0.75}).

**Validation for Pascal VOC validation set**

You can modify *'--model_pth'*  with the path of model you want to evaluate in *$joint_training/pascal_val.sh*.

```
bash pascal_val.sh
```

**Validation for COCO2017 validation set**

You can modify *'--model_pth'*  with the path of model you want to evaluate in *$joint_training/coco_val.sh*.

```
bash coco_val.sh
```

### Generate coarse pseudo mask

---

Generate coarse pseudo mask for weakly-supervised training set instances in a json format. Similarly, you need to modify *'--model_pth'* in related *.bash* file. And code will print where psedo mask file saves.

**Generate pseudo masks for Augmented Pascal VOC training set**

```
bash pascal_psd_mask.sh
```

**Generate pseudo masks for COCO2017 training set**

```
bash coco_psd_mask.sh
```

## <span id = "5">5. Our models download</span>

Pascal: pascal_epoch40_miou_0.718.pth
COCO: COCO_epoch_50_miou0.717.pth

* [BaiDuYun](https://pan.baidu.com/s/1BGyljK-0WthPWur3x73FGw), Access code：2wyu
* [GoogleDrive](https://drive.google.com/drive/folders/12qjGTBzTgehf_5GNF5ph0Rdm3o1xfISt?usp=sharing)

## <span id = "6">6. Some supplements</span>

* Joint training part codes refer codes of [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation), [BBTP](https://github.com/chengchunhsu/WSIS_BBTP).
* Follows U^2 Net, we horizontal flip images of DUT-TR-single and DUT-TR-single-new offline and don't adopt random horizontal flip in data augmentation.
* For COCO training, we adopt a sample strategy which brings about 1 percent promotion in *miou^* *. Trough calulating ground truth boxes' overlaps in a COCO image, we find some COCO instance with serve occlusion. In training stage, we will randomly drop these instances.
