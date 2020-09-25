# Panoptic-DeepLab Model Zoo and Baselines

### Introduction
This file documents a large collection of baselines for Panoptic-DeepLab. We are planning to provide results Panoptic-DeepLab with different backbones as well as projects that are built on Panoptic-DeepLab.

Currently we only have results on the Cityscapes panoptic segmentation benchmark, results on other datasets like COCO and Mapillary Vistas are still under development.

You are wellcome to put results of your model in this model zoo if it is based on Panoptic-DeepLab (e.g. new backbone, new head design, etc).


### Cityscapes baselines
##### ResNet
By default, models are trained with a batch size of 8 with 8 GPUs. You can also train
it with batchsize of 4 using 4 GPUs (without changing learning rate and total number of iterations).

We use the [TorchVision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
ResNet implementation, which is not exactly the same as the 
[TensorFlow](https://github.com/tensorflow/models/blob/master/research/deeplab/core/resnet_v1_beta.py) 
ResNet implementation.

For ResNet-50 model, 11G memory should be enough. You will need larger memory for models larger than R50.
| Name    | PQ   | SQ   | RQ   | AP   | mIoU | Model |
| ------- | ---- | ---- | ---- | ---- | ---- | ----- |
| [R50-os32](configs/panoptic_deeplab_R50_os32_cityscapes.yaml)| 59.8 | 80.0 | 73.5 | 26.9 / 28.3 | 78.6 | [Google Drive](https://drive.google.com/file/d/1IhZXtLpVkzhH4S2k27zARM8kUI7G6Hfn/view?usp=sharing) |
| [R101-os32](configs/panoptic_deeplab_R101_os32_cityscapes.yaml)| 60.3 | 80.8 | 73.6 | 27.7 / 30.2 | 78.4 | [Google Drive](https://drive.google.com/file/d/1I26-bTW55crVLqCFB4lhKdIzz8y3X8qR/view?usp=sharing) |
| [X101-32x8d-os32](configs/panoptic_deeplab_X101_32x8d_os32_cityscapes.yaml)| 61.4 | 80.8 | 74.9 | 28.9 / 30.3 | 79.6 | [Google Drive](https://drive.google.com/file/d/10u5w8dbHysSI1HMbfLLuCMV8kpyeDMcO/view?usp=sharing) |

Note:
- R50/R101: ResNet-50 and ResNet-101
- X101: ResNext-101
- 2 AP numbers refer to different ways to calculate condidence score: [semantic](https://github.com/bowenc0221/panoptic-deeplab/blob/9225f83cba48985263775635f0805f482de6aeeb/segmentation/model/post_processing/evaluation_format.py#L49) / [semantic x instance](https://github.com/bowenc0221/panoptic-deeplab/blob/9225f83cba48985263775635f0805f482de6aeeb/segmentation/model/post_processing/evaluation_format.py#L53)



##### Xception models
| Name    | PQ   | SQ   | RQ   | AP   | mIoU | Model |
| ------- | ---- | ---- | ---- | ---- | ---- | ----- |
| [X65-os32](configs/panoptic_deeplab_X65_os32_cityscapes_lr_x10.yaml)| 61.8 | 81.1 | 75.2 | 30.9 / 31.7 | 79.6 | [Google Drive](https://drive.google.com/file/d/1TjN5gUyzA8Q2HqnP6jBu62GRTyAntQeQ/view?usp=sharing) |

Note:
- X65: Xception-65
- 2 AP numbers refer to different ways to calculate condidence score: [semantic](https://github.com/bowenc0221/panoptic-deeplab/blob/9225f83cba48985263775635f0805f482de6aeeb/segmentation/model/post_processing/evaluation_format.py#L49) / [semantic x instance](https://github.com/bowenc0221/panoptic-deeplab/blob/9225f83cba48985263775635f0805f482de6aeeb/segmentation/model/post_processing/evaluation_format.py#L53)




##### Mobile models
| Name    | PQ   | SQ   | RQ   | AP   | mIoU | Model |
| ------- | ---- | ---- | ---- | ---- | ---- | ----- |
| [MBNV2-1.0-os32](configs/panoptic_deeplab_MBNV2_100_os32_cityscapes_bs16_lr_x2.yaml)| 55.1 | 79.4 | 68.0 | 20.6 / 23.3 | 75.8 | [Google Drive](https://drive.google.com/file/d/1E5wsJuYIjKRt1YQm_lV-bDdyL_1UCaDR/view?usp=sharing) |

Note:
- MBNV2-1.0: MobileNetV2 with width multiplier 1.0
- 2 AP numbers refer to different ways to calculate condidence score: [semantic](https://github.com/bowenc0221/panoptic-deeplab/blob/9225f83cba48985263775635f0805f482de6aeeb/segmentation/model/post_processing/evaluation_format.py#L49) / [semantic x instance](https://github.com/bowenc0221/panoptic-deeplab/blob/9225f83cba48985263775635f0805f482de6aeeb/segmentation/model/post_processing/evaluation_format.py#L53)



##### HRNet series (Under Progress)

We trained HRNet-W48 baseline on Cityscapes with 8 Tesla V100 GPUs, with a batchsize 8.
We use the [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) HRNet implementation.
All the results are under progress and we are making efforts to finetune various hyperparameters to achieve better performance. 

We use Pytorch1.5 in all of our experiments. The docker is [rainbowsecret/pytorch1.5:latest](https://hub.docker.com/repository/docker/rainbowsecret/pytorch1.5).

| Name    | PQ   | SQ   | RQ   | AP   | mIoU | Model |
| ------- | ---- | ---- | ---- | ---- | ---- | ----- |
| [H48-os4](configs/panoptic_deeplab_H48_os4_cityscapes.yaml)| 63.4  |  81.5  |  76.7 | 29.9 / 29.6 | 80.9 |  [Google Drive](https://drive.google.com/drive/folders/1bJLyZkKsharpGykxjR7hmb6yzp8nmxMj?usp=sharing) |

Note:
- H48: HRNet with width 48
- 2 AP numbers refer to different ways to calculate condidence score: [semantic](https://github.com/bowenc0221/panoptic-deeplab/blob/9225f83cba48985263775635f0805f482de6aeeb/segmentation/model/post_processing/evaluation_format.py#L49) / [semantic x instance](https://github.com/bowenc0221/panoptic-deeplab/blob/9225f83cba48985263775635f0805f482de6aeeb/segmentation/model/post_processing/evaluation_format.py#L53)

