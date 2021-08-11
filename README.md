# Unfolding the Alternating Optimization for blind super resolution 

## Introduction
This repo is for [OpenMMLab Algorithm Ecological Challenge](https://openmmlab.com/competitions/algorithm-2021), and paper is [Unfolding the Alternating Optimization for blind super resolution](https://arxiv.org/pdf/2010.02631v4.pdf)

## Training
We add some codes based on [openmmlab/mmediting](https://github.com/open-mmlab/mmediting). You have two methods to train:
- pre-generate all training set by using our [preprocess_div2k_dataset.py](https://github.com/wileechou/dan_mmediting/blob/master/tools/data/blind-super-resolution/div2k/preprocess_div2k_dataset.py) and 
use our dataset class [bsr_folder_dataset.py](https://github.com/wileechou/dan_mmediting/blob/master/mmedit/datasets/bsr_folder_dataset.py)
- generate low-quality images and kernels during training by defining the training pipeline in configuration file. We add a degradation class in [augmentaion.py](https://github.com/wileechou/dan_mmediting/blob/master/mmedit/datasets/pipelines/augmentation.py)

You can just use two config files：
 - [dan_div2k_x4_gt_only.py](https://github.com/wileechou/dan_mmediting/blob/master/configs/restorers/DAN/dan_div2k_x4_gt_only.py). Remember replace `augmentation.py` with ours in mmediting
 - [DAN_DIV2K_x4_v2.py](https://github.com/wileechou/dan_mmediting/blob/master/configs/restorers/DAN/DAN_DIV2K_x4_v2.py). 

## Evaluation
The evaluation for two methods are the same because we define same pipelines. Just be careful with datasetloader we use.
Model weights: [Baidu:eqp3](https://pan.baidu.com/s/1NfURCcWRMMb517r8gn2DZQ)
