# Panoptic-DeepLab (CVPR 2020)

Panoptic-DeepLab is a state-of-the-art bottom-up method for panoptic segmentation, 
where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to 
every pixel in the input image as well as instance labels (e.g. an id of 1, 2, 3, 
etc) to pixels belonging to thing classes. 

![Illustrating of Panoptic-DeepLab](/docs/panoptic_deeplab.png)

This is the **PyTorch re-implementation** of our CVPR2020 paper: 
[Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation](https://arxiv.org/abs/1911.10194).

## News
* [2020/07/21] Check this [Google AI Blog](https://ai.googleblog.com/2020/07/improving-holistic-scene-understanding.html) for Panoptic-DeepLab.
* [2020/07/01] More Cityscapes pre-trained backbones in model zoo.
* [2020/06/30] Panoptic-DeepLab now supports [HRNet](https://github.com/HRNet), using HRNet-w48 backbone achieves 63.4% PQ on Cityscapes. Thanks to @PkuRainBow.

## Community contribution
If you are interested in contributing to improve this PyTorch implementation of Panoptic-DeepLab, here is a list of TODO tasks.
You can claim the task by opening an issue and we can discuss futher.

Features:
- [X] Add a demo code that takes a single image as input and saves visualization outputs (#22).
- [ ] Support COCO and Mapillary Vistas models.
- [ ] Support multi-node distributed training.
- [ ] Support mixed precision (fp16) training.
- [ ] Optimize post-processing (make it parallel).
- [ ] Reproduce Xception results.

Debugging:
- [ ] AP number is a little bit lower than our original implementation.
- [ ] Currently there are some problem training ResNet with output stride = 16 (it gets much lower PQ).

## Disclaimer
* This is a **re-implementation** of Panoptic-DeepLab, it is not guaranteed to reproduce all numbers in the paper, please refer to the
original numbers from [Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation](https://arxiv.org/abs/1911.10194)
when making comparison.

## What's New
* We release a detailed [technical report](/docs/tech_report.pdf) with implementation details 
and supplementary analysis on Panoptic-DeepLab. In particular, we find center prediction is almost perfect and the bottleneck of 
bottom-up method still lies in semantic segmentation
* It is powered by the [PyTorch](https://pytorch.org) deep learning framework.
* Can be trained even on 4 1080TI GPUs (no need for 32 TPUs!).

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md).

## Model Zoo

See [MODEL_ZOO.md](MODEL_ZOO.md).

## Changelog

See [changelog](/docs/changelog.md)

## Citing Panoptic-DeepLab

If you find this code helpful in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```BibTeX
@inproceedings{cheng2020panoptic,
  title={Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation},
  author={Cheng, Bowen and Collins, Maxwell D and Zhu, Yukun and Liu, Ting and Huang, Thomas S and Adam, Hartwig and Chen, Liang-Chieh},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{cheng2019panoptic,
  title={Panoptic-DeepLab},
  author={Cheng, Bowen and Collins, Maxwell D and Zhu, Yukun and Liu, Ting and Huang, Thomas S and Adam, Hartwig and Chen, Liang-Chieh},
  booktitle={ICCV COCO + Mapillary Joint Recognition Challenge Workshop},
  year={2019}
}
```

If you use the HRNet backbone, please consider citing
```
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI},
  year={2019}
}
```

## Acknowledgements
We have used utility functions from other wonderful open-source projects, we would espeicially thank the authors of:
- [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [TorchVision](https://github.com/pytorch/vision)

## Contact
[Bowen Cheng](https://bowenc0221.github.io/) (bcheng9 AT illinois DOT edu)
