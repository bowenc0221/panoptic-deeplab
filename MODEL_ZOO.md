# Model Zoo and Baselines

### Introduction
We trained ResNet-50 baseline on Cityscapes with 4 1080TI GPUs, with a batchsize of 4.
We use the [TorchVision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
ResNet implementation, which is not exactly the same as the 
[TensorFlow](https://github.com/tensorflow/models/blob/master/research/deeplab/core/resnet_v1_beta.py) 
ResNet implementation.
We use the [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) HRNet implementation.

### Cityscapes baselines (Under Progress)

| Name    | PQ   | SQ   | RQ   | AP   | mIoU | Model |
| ------- | ---- | ---- | ---- | ---- | ---- | ----- |
| [R50-os32](configs/panoptic_deeplab_R50_os32_cityscapes.yaml)| 59.8 | 80.0 | 73.5 | 26.9 | 78.6 | [Google Drive](https://drive.google.com/file/d/1IhZXtLpVkzhH4S2k27zARM8kUI7G6Hfn/view?usp=sharing) |


###### HRNet series (Under Progress)

We trained HRNet-W48 baseline on Cityscapes with 8 V100 GPUs, with a batchsize 8.
We use the [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) HRNet implementation.
All the results are under progress and we are making efforts to finetune various hyperparameters to achieve better performance. 

We use Pytorch1.5 in all of our experiments. The docker is [rainbowsecret/pytorch1.5:latest](https://hub.docker.com/repository/docker/rainbowsecret/pytorch1.5).

| Name    | PQ   | SQ   | RQ   | AP   | mIoU | Model |
| ------- | ---- | ---- | ---- | ---- | ---- | ----- |
| [H48-os4](panoptic_deeplab_H48_os4_cityscapes.yaml)| 63.4  |  81.5  |  76.7 | 29.9 | 80.9 |  [Google Drive](https://drive.google.com/drive/folders/1bJLyZkKsharpGykxjR7hmb6yzp8nmxMj?usp=sharing) |

