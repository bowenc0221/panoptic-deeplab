# ------------------------------------------------------------------------------
# Demo code.
# Example command:
# python tools/demo.py --cfg PATH_TO_CONFIG_FILE \
#   --input-files PATH_TO_INPUT_FILES \
#   --output-dir PATH_TO_OUTPUT_DIR
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import argparse
import cv2
import os
import pprint
import logging
import time
import glob

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import _init_paths
from fvcore.common.file_io import PathManager
from segmentation.config import config, update_config
from segmentation.utils.logger import setup_logger
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.utils import save_debug_images
from segmentation.model.post_processing import get_semantic_segmentation, get_panoptic_segmentation
from segmentation.utils import save_annotation, save_instance_annotation, save_panoptic_annotation
import segmentation.data.transforms.transforms as T
from segmentation.utils import AverageMeter
from segmentation.data import build_test_loader_from_cfg


def read_image(file_name, format=None):
    image = Image.open(file_name)

    # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    return image


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--input-files',
                        help='input files, could be image, image list or video',
                        required=True,
                        type=str)
    parser.add_argument('--output-dir',
                        help='output directory',
                        required=True,
                        type=str)
    parser.add_argument('--extension',
                        help='file extension if input is image list',
                        default='.png',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger = logging.getLogger('demo')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=args.output_dir, name='demo')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.TEST.GPUS)
    if len(gpus) > 1:
        raise ValueError('Test only supports single core.')
    device = torch.device('cuda:{}'.format(gpus[0]))

    # build model
    model = build_segmentation_model_from_cfg(config)

    # Change ASPP image pooling
    # output_stride = 2 ** (5 - sum(config.MODEL.BACKBONE.DILATION))
    # train_crop_h, train_crop_w = config.TEST.CROP_SIZE
    # scale = 1. / output_stride
    # pool_h = int((float(train_crop_h) - 1.0) * scale + 1.0)
    # pool_w = int((float(train_crop_w) - 1.0) * scale + 1.0)

    # model.set_image_pooling((pool_h, pool_w))

    logger.info("Model:\n{}".format(model))
    model = model.to(device)

    # build data_loader
    # TODO: still need it for thing_list
    data_loader = build_test_loader_from_cfg(config)

    # load model
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(config.OUTPUT_DIR, 'final_state.pth')

    if os.path.isfile(model_state_file):
        model_weights = torch.load(model_state_file)
        if 'state_dict' in model_weights.keys():
            model_weights = model_weights['state_dict']
            logger.info('Evaluating a intermediate checkpoint.')
        model.load_state_dict(model_weights, strict=True)
        logger.info('Test model loaded from {}'.format(model_state_file))
    else:
        if not config.DEBUG.DEBUG:
            raise ValueError('Cannot find test model.')

    # load images
    input_list = []
    if os.path.exists(args.input_files):
        if os.path.isfile(args.input_files):
            # inference on a single file, extract extension
            ext = os.path.splitext(os.path.basename(args.input_files))[1]
            if ext in ['.png', '.jpg', '.jpeg']:
                # image file
                input_list.append(args.input_files)
            elif ext in ['.mpeg']:
                # video file
                # TODO: decode video and convert to image list
                raise NotImplementedError("Inference on video is not supported yet.")
            else:
                raise ValueError("Unsupported extension: {}.".format(ext))
        else:
            # inference on a directory
            for fname in glob.glob(os.path.join(args.input_files, '*' + args.extension)):
                input_list.append(fname)
    else:
        raise ValueError('Input file or directory does not exists: {}'.format(args.input_files))

    if isinstance(input_list[0], str):
        logger.info("Inference on images")
        logger.info(input_list)
    else:
        logger.info("Inference on video")

    # dir to save intermediate raw outputs
    raw_out_dir = os.path.join(args.output_dir, 'raw')
    PathManager.mkdirs(raw_out_dir)

    # dir to save semantic outputs
    semantic_out_dir = os.path.join(args.output_dir, 'semantic')
    PathManager.mkdirs(semantic_out_dir)

    # dir to save instance outputs
    instance_out_dir = os.path.join(args.output_dir, 'instance')
    PathManager.mkdirs(instance_out_dir)

    # dir to save panoptic outputs
    panoptic_out_dir = os.path.join(args.output_dir, 'panoptic')
    PathManager.mkdirs(panoptic_out_dir)

    # Test loop
    model.eval()

    # build image demo transform
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                config.DATASET.MEAN,
                config.DATASET.STD
            )
        ]
    )

    # logger.info("Warmup GPU.")
    # with torch.no_grad():
    #     for _ in range(10):
    #         if isinstance(input_list[0], str):
    #             # load image
    #             image = read_image(input_list[0], 'RGB')
    #         else:
    #             NotImplementedError("Inference on video is not supported yet.")
            
    #         image, _ = transforms(image, None)
    #         model(image.unsqueeze(0).to(device))
    #         torch.cuda.synchronize(device)

    net_time = AverageMeter()
    post_time = AverageMeter()
    try:
        with torch.no_grad():
            for i, fname in enumerate(input_list):
                if isinstance(fname, str):
                    # load image
                    image = read_image(fname, 'RGB')
                else:
                    NotImplementedError("Inference on video is not supported yet.")
                
                # pad image
                raw_shape = image.shape[:2]
                raw_h = raw_shape[0]
                raw_w = raw_shape[1]
                new_h = (raw_h + 31) // 32 * 32 + 1
                new_w = (raw_w + 31) // 32 * 32 + 1
                input_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                input_image[:, :] = config.DATASET.MEAN
                input_image[:raw_h, :raw_w, :] = image

                image, _ = transforms(input_image, None)
                image = image.unsqueeze(0).to(device)

                # network
                start_time = time.time()
                out_dict = model(image)
                torch.cuda.synchronize(device)
                net_time.update(time.time() - start_time)

                # post-processing
                start_time = time.time()
                semantic_pred = get_semantic_segmentation(out_dict['semantic'])

                panoptic_pred, center_pred = get_panoptic_segmentation(
                    semantic_pred,
                    out_dict['center'],
                    out_dict['offset'],
                    thing_list=data_loader.dataset.thing_list,
                    label_divisor=data_loader.dataset.label_divisor,
                    stuff_area=config.POST_PROCESSING.STUFF_AREA,
                    void_label=(
                            data_loader.dataset.label_divisor *
                            data_loader.dataset.ignore_label),
                    threshold=config.POST_PROCESSING.CENTER_THRESHOLD,
                    nms_kernel=config.POST_PROCESSING.NMS_KERNEL,
                    top_k=config.POST_PROCESSING.TOP_K_INSTANCE,
                    foreground_mask=None)
                torch.cuda.synchronize(device)
                post_time.update(time.time() - start_time)

                logger.info('[{}/{}]\t'
                            'Network Time: {net_time.val:.3f}s ({net_time.avg:.3f}s)\t'
                            'Post-processing Time: {post_time.val:.3f}s ({post_time.avg:.3f}s)\t'.format(
                             i, len(input_list), net_time=net_time, post_time=post_time))
                
                # save predictions
                semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
                panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()

                # crop predictions
                semantic_pred = semantic_pred[:raw_h, :raw_w]
                panoptic_pred = panoptic_pred[:raw_h, :raw_w]

                # Raw outputs
                save_debug_images(
                    dataset=data_loader.dataset,
                    batch_images=image,
                    batch_targets={},
                    batch_outputs=out_dict,
                    out_dir=raw_out_dir,
                    iteration=i,
                    target_keys=[],
                    output_keys=['semantic', 'center', 'offset'],
                    is_train=False,
                )

                save_annotation(semantic_pred, semantic_out_dir, 'semantic_pred_%d' % i,
                                add_colormap=True, colormap=data_loader.dataset.create_label_colormap())
                pan_to_sem = panoptic_pred // data_loader.dataset.label_divisor
                save_annotation(pan_to_sem, semantic_out_dir, 'pan_to_sem_pred_%d' % i,
                                add_colormap=True, colormap=data_loader.dataset.create_label_colormap())
                ins_id = panoptic_pred % data_loader.dataset.label_divisor
                pan_to_ins = panoptic_pred.copy()
                pan_to_ins[ins_id == 0] = 0
                save_instance_annotation(pan_to_ins, instance_out_dir, 'pan_to_ins_pred_%d' % i)
                save_panoptic_annotation(panoptic_pred, panoptic_out_dir, 'panoptic_pred_%d' % i,
                                         label_divisor=data_loader.dataset.label_divisor,
                                         colormap=data_loader.dataset.create_label_colormap())
    except Exception:
        logger.exception("Exception during demo:")
        raise
    finally:
        logger.info("Demo finished.")


if __name__ == '__main__':
    main()
