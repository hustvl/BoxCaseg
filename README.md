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
* [BaiduCloud](https://pan.baidu.com/s/1lZpXdzz4U7BB-Kf58L2A7g) The password is 'yint'.
* [GoogleDrive](https://drive.google.com/drive/folders/12qjGTBzTgehf_5GNF5ph0Rdm3o1xfISt?usp=sharing)

### Training 
1. After downloading the dataset, put them in the specific folder. Then, run

   More detailed instructions are provided in the `jointraining`.

2. Generate the proxy masks for box-supervised dataset.

3. Retrain a Mask R-CNN.


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
