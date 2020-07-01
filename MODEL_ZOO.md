# Model Zoo and Baselines

### Introduction
We trained ResNet-50 baseline on Cityscapes with 4 1080TI GPUs, with a batchsize of 4.  
We use the [TorchVision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
ResNet implementation, which is not exactly the same as the 
[TensorFlow](https://github.com/tensorflow/models/blob/master/research/deeplab/core/resnet_v1_beta.py) 
ResNet implementation.

### Cityscapes baselines
| Name    | PQ   | SQ   | RQ   | AP   | mIoU | Model |
| ------- | ---- | ---- | ---- | ---- | ---- | ----- |
| [R50-os32](configs/panoptic_deeplab_R50_os32_cityscapes.yaml)| 59.8 | 80.0 | 73.5 | 26.9 | 78.6 | [Google Drive](https://drive.google.com/file/d/1IhZXtLpVkzhH4S2k27zARM8kUI7G6Hfn/view?usp=sharing) |
| [R101-os32](configs/panoptic_deeplab_R101_os32_cityscapes.yaml)| 60.3 | 80.8 | 73.6 | 27.7 | 78.4 | [Google Drive]() |
| [X101-32x8d-os32](configs/panoptic_deeplab_X101_32x8d_os32_cityscapes.yaml)| 61.4 | 80.8 | 74.9 | 28.9 | 79.6 | [Google Drive]() |
