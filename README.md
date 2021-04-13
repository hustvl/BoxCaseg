# Weakly-supervised Instance Segmentation via Class-agnostic Learning with Salient Images.(CVPR 2021)

This is the official code of the paper [Weakly-supervised Instance Segmentation via Class-agnostic Learning with Salient Images](https://arxiv.org/pdf/2104.01526v1.pdf), published in CVPR 2021, by [Xinggang Wang](https://xinggangw.info), Jiapei Feng, Bin Hu, Qi Ding, Longjin Ran, Xiaoxin Chen, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/).


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Introduction">Introduction</a>
    </li>
    <li>
      <a href="#prerequisites">Prerequisites</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#Citation">Citation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- INTRODUCTION -->
## Introduction
<p align="center"><img src="figures/pipeline.png" alt="pipeline" width="90%"></p>

Weakly-supervised instance segmentation uses coarser annotations to acquire a high-precision instance segmentation model, such as bounding boxes. This "box-to-seg" process is a class-agnostic process. Our motivation is to learn a model that achieves generic class-agnostic segmentation. 

The training process is divided into three steps. Firstly, we obtain a class-agnostic segmentation model through the joint training of box-supervised dataset and salient object segmentation dataset. Secondly, we use the class-agnostic segmentation model to produce high-quality predictions for training instances. Those segmentation predictions are merged to generate proxy instance masks for training images. Finally, we re-train a Mask R-CNN with proxy masks.

<!-- PREREQUISITES -->
## Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Python>=3.6, PyTorch
* Augmented PASCAL VOC 2012 Dataset (10582 images for training)
* DUT-TR-Single Dataset

<!-- USAGE -->
## Usage
Here, we provide the DUT-TR-Single datasets, the cocostyle annotations of PASCAL VOC, our pre-trained models and proxy masks for training set.
* [BaiduCloud](https://pan.baidu.com/s/1lZpXdzz4U7BB-Kf58L2A7g) The access code is 'yint'.
* [GoogleDrive](https://drive.google.com/drive/folders/12qjGTBzTgehf_5GNF5ph0Rdm3o1xfISt?usp=sharing)

### Training 
* After downloading the dataset, put them in the specific folder. Then, `cd jointraining` and run the following command to do joint training:
```
    bash train_pd.sh
```
for evaluation:
```
    bash pascal_val.sh
```
for predicting segmentation maps of the training instances:
```
    bash pascal_psd_mask.sh
```
More detailed instructions are provided in the `jointraining`.

* Generate the proxy masks for box-supervised dataset. Run the following command: 
```
    cd proxy_mask
    python pascal_proxy_mask_generate.py --gt-path training_set_boundingbox_cocostyle_json --seg-pred predicted_results
```
* Retrain a Mask R-CNN. We use the [Mask R-CNN](https://github.com/facebookresearch/maskrcnn-benchmark) as our instance segmentation framework and we modify two file include `../maskrcnn_benchmark/data/datasets/coco.py` and `../maskrcnn_benchmark/modeling/roi_heads/mask_head/loss.py`.
 
 Run the following command: 
```
    cd retrain/maskrcnn-benchmark/
    python -m torch.distributed.launch --nproc_per_node=2 ./tools/train_net.py --config-file e2e_mask_rcnn_R_101_FPN_4x_voc_aug_cocostyle.yaml
```
Check [INSTALL.md](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md) for installation instructions.

<!-- CITATION -->
## Citation
If you find the code useful in your research, please consider citing:
```BibTeX
@inproceedings{wang2021boxcaseg,
  title     =  {Weakly-supervised Instance Segmentation via Class-agnostic Learning with Salient Images},
  author    =  {Wang, Xinggang and Feng, Jiapei and Hu, Bin and Ding, Qi and Ran, Longjin and Chen, Xiaoxin and Liu, Wenyu},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2021}
}
```

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* We thank NSFC, Zhejiang Laboratory and VIVO Inc for their support to this project.
* The code is borrowed from [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation), [BBTP](https://github.com/chengchunhsu/WSIS_BBTP) and [Mask R-CNN](https://github.com/facebookresearch/maskrcnn-benchmark).
