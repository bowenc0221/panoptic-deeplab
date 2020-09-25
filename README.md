# Panoptic-DeepLab (CVPR 2020)

Panoptic-DeepLab is a state-of-the-art bottom-up method for panoptic segmentation, 
where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to 
every pixel in the input image as well as instance labels (e.g. an id of 1, 2, 3, 
etc) to pixels belonging to thing classes. 

![Illustrating of Panoptic-DeepLab](/docs/panoptic_deeplab.png)

This is the **PyTorch re-implementation** of our CVPR2020 paper: 
[Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation](https://arxiv.org/abs/1911.10194).

## News
* [2020/09/24] I have implemented both [DeepLab](https://github.com/facebookresearch/detectron2/tree/master/projects/DeepLab) and [Panoptic-DeepLab](https://github.com/facebookresearch/detectron2/tree/master/projects/Panoptic-DeepLab) in the official [Detectron2](https://github.com/facebookresearch/detectron2), the implementation in the repo will be deprecated and I will mainly maintain the Detectron2 version. However, this repo still support different backbones for the Detectron2 Panoptic-DeepLab.
* [2020/07/21] Check this [Google AI Blog](https://ai.googleblog.com/2020/07/improving-holistic-scene-understanding.html) for Panoptic-DeepLab.
* [2020/07/01] More Cityscapes pre-trained backbones in model zoo (MobileNet and Xception are supported).
* [2020/06/30] Panoptic-DeepLab now supports [HRNet](https://github.com/HRNet), using HRNet-w48 backbone achieves 63.4% PQ on Cityscapes. Thanks to @PkuRainBow.

## Results

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Output<br/>resolution</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left">Panoptic-DeepLab</td>
<td align="center">R52-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 60.3 </td>
<td align="center"> 81.5 </td>
<td align="center"> 72.9 </td>
<td align="center"> 78.2 </td>
<td align="center"> 33.2 </td>
<td align="center"><a href="
">model</a></td>
</tr>
 <tr><td align="left"><a href="tools_d2/config/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024.yaml">Panoptic-DeepLab</a></td>
<td align="center">X65-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 62.6 </td>
<td align="center"> 81.5 </td>
<td align="center"> 75.7 </td>
<td align="center"> 79.4 </td>
<td align="center"> 32.8 </td>
<td align="center"><a href="
">model</a></td>
</tr>
 <tr><td align="left"><a href="tools_d2/config/Cityscapes-PanopticSegmentation/panoptic_deeplab_H_48_os16_mg124_poly_90k_bs32_crop_512_1024.yaml">Panoptic-DeepLab</a></td>
<td align="center">HRNet-48</td>
<td align="center">1024&times;2048</td>
<td align="center"> 63.3 </td>
<td align="center"> 82.2 </td>
<td align="center"> 76.0 </td>
<td align="center"> 80.3 </td>
<td align="center"> 35.9 </td>
<td align="center"><a href="
">model</a></td>
</tr>
</tbody></table>

## Disclaimer
* The implementation in this repo will be depracated, please refer to my [Detectron2 implementation](https://github.com/facebookresearch/detectron2/tree/master/projects/Panoptic-DeepLab) which gives slightly better results.
* This is a **re-implementation** of Panoptic-DeepLab, it is not guaranteed to reproduce all numbers in the paper, please refer to the
original numbers from [Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation](https://arxiv.org/abs/1911.10194)
when making comparison.

## What's New
* We release a detailed [technical report](/docs/tech_report.pdf) with implementation details 
and supplementary analysis on Panoptic-DeepLab. In particular, we find center prediction is almost perfect and the bottleneck of 
bottom-up method still lies in semantic segmentation
* It is powered by the [PyTorch](https://pytorch.org) deep learning framework.
* Can be trained even on 4 1080TI GPUs (no need for 32 TPUs!).

## How to use
We suggest using the Detectron2 implementation. You can either use it directly from the [Detectron2 projects](https://github.com/facebookresearch/detectron2/tree/master/projects/Panoptic-DeepLab) or use it from this repo from [tools_d2/README.md](/tools_d2/README.md).

The differences are, official Detectron2 implementation only supports ResNet or ResNeXt as the backbone. This repo gives you an example of how to use your a custom backbone within Detectron2.

Note:
* Please check the usage of this code in [tools_d2/README.md](/tools_d2/README.md).
* If you are still interested in the old code, please check [tools/README.md](/tools/README.md).

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

If you use the Xception backbone, please consider citing
```BibTeX
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}

@inproceedings{qi2017deformable,
  title={Deformable convolutional networks--coco detection and segmentation challenge 2017 entry},
  author={Qi, Haozhi and Zhang, Zheng and Xiao, Bin and Hu, Han and Cheng, Bowen and Wei, Yichen and Dai, Jifeng},
  booktitle={ICCV COCO Challenge Workshop},
  year={2017}
}
```

If you use the HRNet backbone, please consider citing
```BibTeX
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
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
