# Model Zoo and Baselines

### Introduction
We use the [TorchVision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
ResNet implementation, which is not exactly the same as the 
[TensorFlow](https://github.com/tensorflow/models/blob/master/research/deeplab/core/resnet_v1_beta.py) 
ResNet implementation.

### Cityscapes baselines
##### ResNet
By default, models are trained with a batch size of 8 with 8 GPUs. You can also train
it with batchsize of 4 using 4 GPUs (without changing learning rate and total number of iterations).

For ResNet-50 model, 11G memory should be enough. You will need larger memory for models larger than R50.

| Name    | PQ   | SQ   | RQ   | AP   | mIoU | Model |
| ------- | ---- | ---- | ---- | ---- | ---- | ----- |
| [R50-os32](configs/panoptic_deeplab_R50_os32_cityscapes.yaml)| 59.8 | 80.0 | 73.5 | 26.9 | 78.6 | [Google Drive](https://drive.google.com/file/d/1IhZXtLpVkzhH4S2k27zARM8kUI7G6Hfn/view?usp=sharing) |
| [R101-os32](configs/panoptic_deeplab_R101_os32_cityscapes.yaml)| 60.3 | 80.8 | 73.6 | 27.7 | 78.4 | [Google Drive](https://drive.google.com/file/d/1I26-bTW55crVLqCFB4lhKdIzz8y3X8qR/view?usp=sharing) |
| [X101-32x8d-os32](configs/panoptic_deeplab_X101_32x8d_os32_cityscapes.yaml)| 61.4 | 80.8 | 74.9 | 28.9 | 79.6 | [Google Drive](https://drive.google.com/file/d/10u5w8dbHysSI1HMbfLLuCMV8kpyeDMcO/view?usp=sharing) |



##### HRNet series (Under Progress)

We trained HRNet-W48 baseline on Cityscapes with 8 Tesla V100 GPUs, with a batchsize 8.
We use the [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) HRNet implementation.
All the results are under progress and we are making efforts to finetune various hyperparameters to achieve better performance. 

We use Pytorch1.5 in all of our experiments. The docker is [rainbowsecret/pytorch1.5:latest](https://hub.docker.com/repository/docker/rainbowsecret/pytorch1.5).

| Name    | PQ   | SQ   | RQ   | AP   | mIoU | Model |
| ------- | ---- | ---- | ---- | ---- | ---- | ----- |
| [H48-os4](configs/panoptic_deeplab_H48_os4_cityscapes.yaml)| 63.4  |  81.5  |  76.7 | 29.9 | 80.9 |  [Google Drive](https://drive.google.com/drive/folders/1bJLyZkKsharpGykxjR7hmb6yzp8nmxMj?usp=sharing) |

