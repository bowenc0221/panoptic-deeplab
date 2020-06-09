# Panoptic-DeepLab (CVPR 2020)

Panoptic-DeepLab is a state-of-the-art bottom-up method for panoptic segmentation, 
where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to 
every pixel in the input image as well as instance labels (e.g. an id of 1, 2, 3, 
etc) to pixels belonging to thing classes. 

This is the **PyTorch re-implementation** of our CVPR2020 paper: 
[Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation](https://arxiv.org/abs/1911.10194).

## Disclaimer
* This is a **re-implementation** of Panoptic-DeepLab, it is not guaranteed to reproduce all numbers in the paper, please refer to the
original numbers from [Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation](https://arxiv.org/abs/1911.10194)
when making comparison.

## What's New
* It is powered by the [PyTorch](https://pytorch.org) deep learning framework.
* Can be trained even on 4 1080TI GPUs (no need for 32 TPUs!).

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md).

## Model Zoo

See [MODEL_ZOO.md](MODEL_ZOO.md).

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

## Acknowledgements
We have used utility functions from other wonderful open-source projects, we would espeicially thank the authors of:
- [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [TorchVision](https://github.com/pytorch/vision)

## Contact
[Bowen Cheng](https://bowenc0221.github.io/) (bcheng9 AT illinois DOT edu)
