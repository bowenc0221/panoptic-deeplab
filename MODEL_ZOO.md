# Model Zoo and Baselines

### Introduction
We trained ResNet-50 baseline on Cityscapes with 4 1080TI GPUs, with a batchsize of 4.  
We use the [TorchVision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
ResNet implementation, which is not exactly the same as the 
[TensorFlow](https://github.com/tensorflow/models/blob/master/research/deeplab/core/resnet_v1_beta.py) 
ResNet implementation.

### Cityscapes baselines (Under Progress)

All the results are under progress and we are making efforts to finetune various hyperparameters to achieve better performance.

| Name    | PQ   | SQ   | RQ   | AP   | mIoU | Model |
| ------- | ---- | ---- | ---- | ---- | ---- | ----- |
| [R50-B8-O32-90K](configs/panoptic_deeplab_R50_B8_O32_90K_cityscapes.yaml)| 59.3 |  80.3 |  72.7 | 25.4 | 78.6 | N/A |
| [R50-B8-O32-120K](configs/panoptic_deeplab_R50_B8_O32_120K_cityscapes.yaml)| 58.7 |  80.6 |  71.7 | 24.1 | 78.9 | N/A |
| [R101-B8-O32-90K](configs/panoptic_deeplab_R101_B8_O32_120K_cityscapes.yaml)| - |  - |  - | - | - | N/A |
| [R101-B8-O32-120K](configs/panoptic_deeplab_R101_B8_O32_120K_cityscapes.yaml)| 59.9 |  80.6 |  73.2 | 27.3 | 78.3 | N/A |
| [H48-B8-O4-90K](configs/panoptic_deeplab_R101_B8_O32_120K_cityscapes.yaml)| - |  - |  - | - | - | N/A |
| [H48-B8-O4-120K](configs/panoptic_deeplab_R101_B8_O32_120K_cityscapes.yaml)| - |  - |  - | - | - | N/A |