from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .conv_module import stacked_conv
from .panoptic_deeplab import SinglePanopticDeepLabDecoder, SinglePanopticDeepLabHead


__all__ = ["PanopticDeepLabDecoderV2"]


class PanopticDeepLabDecoderV2(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, num_classes, **kwargs):
        super(PanopticDeepLabDecoderV2, self).__init__()
        # Build semantic decoder
        self.semantic_decoder = SinglePanopticDeepLabDecoder(in_channels, feature_key, low_level_channels,
                                                             low_level_key, low_level_channels_project,
                                                             decoder_channels, atrous_rates)
        self.semantic_head = SinglePanopticDeepLabHead(decoder_channels, [num_classes], ['semantic'])
        # Build instance decoder
        self.instance_decoder = None
        self.instance_head = None
        self.foreground_head = None
        if kwargs.get('has_instance', False):
            instance_decoder_kwargs = dict(
                in_channels=in_channels,
                feature_key=feature_key,
                low_level_channels=low_level_channels,
                low_level_key=low_level_key,
                low_level_channels_project=kwargs['instance_low_level_channels_project'],
                decoder_channels=kwargs['instance_decoder_channels'],
                atrous_rates=atrous_rates,
                aspp_channels=kwargs['instance_aspp_channels']
            )
            self.instance_decoder = SinglePanopticDeepLabDecoder(**instance_decoder_kwargs)
            instance_head_kwargs = dict(
                decoder_channels=kwargs['instance_decoder_channels'],
                num_classes=kwargs['instance_num_classes'],
                class_key=kwargs['instance_class_key']
            )
            self.instance_head = SinglePanopticDeepLabHead(**instance_head_kwargs)

            if kwargs.get('foreground_seg', False):
                fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                                    conv_type='depthwise_separable_conv')
                self.foreground_arch = kwargs.get('foreground_arch', 'v1')
                if self.foreground_arch == 'v1':
                    # Simply on semantic decoder.
                    self.foreground_fuse = fuse_conv(decoder_channels, decoder_channels)
                    self.foreground_head = SinglePanopticDeepLabHead(decoder_channels, [2], ['foreground'])
                elif self.foreground_arch == 'v2':
                    # Simply on instance decoder.
                    self.foreground_fuse = fuse_conv(kwargs['instance_decoder_channels'], decoder_channels)
                    self.foreground_head = SinglePanopticDeepLabHead(
                        decoder_channels, [2], ['foreground'])
                elif self.foreground_arch == 'v3':
                    # Use concatenated semantic decoder and instance decoder features.
                    self.foreground_fuse = fuse_conv(
                        decoder_channels + kwargs['instance_decoder_channels'], decoder_channels)
                    self.foreground_head = SinglePanopticDeepLabHead(
                        decoder_channels, [2], ['foreground'])
                elif self.foreground_arch == 'v4':
                    # Use semantic prediction.
                    self.foreground_fuse = fuse_conv(num_classes, decoder_channels)
                    self.foreground_head = SinglePanopticDeepLabHead(
                        decoder_channels, [2], ['foreground'])
                elif self.foreground_arch == 'v5':
                    # Use center and offset prediction.
                    self.foreground_fuse = fuse_conv(sum(kwargs['instance_num_classes']), decoder_channels)
                    self.foreground_head = SinglePanopticDeepLabHead(
                        decoder_channels, [2], ['foreground'])
                elif self.foreground_arch == 'v6':
                    # Use semantic, center and offset prediction.
                    self.foreground_fuse = fuse_conv(
                        num_classes + sum(kwargs['instance_num_classes']), decoder_channels)
                    self.foreground_head = SinglePanopticDeepLabHead(
                        decoder_channels, [2], ['foreground'])
                elif self.foreground_arch == 'v7':
                    # Use semantic decoder, instance decoder and semantic prediction.
                    self.foreground_fuse = fuse_conv(
                        decoder_channels + kwargs['instance_decoder_channels'] + num_classes, decoder_channels)
                    self.foreground_head = SinglePanopticDeepLabHead(
                        decoder_channels, [2], ['foreground'])
                else:
                    raise ValueError('Undefined foreground_arch: {}'.format(self.foreground_arch))

    def set_image_pooling(self, pool_size):
        self.semantic_decoder.set_image_pooling(pool_size)
        if self.instance_decoder is not None:
            self.instance_decoder.set_image_pooling(pool_size)

    def forward(self, features):
        pred = OrderedDict()

        # Semantic branch
        semantic_decoder = self.semantic_decoder(features)
        semantic = self.semantic_head(semantic_decoder)
        for key in semantic.keys():
            pred[key] = semantic[key]

        # Instance branch
        if self.instance_decoder is not None:
            instance_decoder = self.instance_decoder(features)
            instance = self.instance_head(instance_decoder)
            for key in instance.keys():
                pred[key] = instance[key]

            # Foreground head
            if self.foreground_head is not None:
                if self.foreground_arch == 'v1':
                    foreground_feature = self.foreground_fuse(semantic_decoder)
                    foreground = self.foreground_head(foreground_feature)
                elif self.foreground_arch == 'v2':
                    foreground_feature = self.foreground_fuse(instance_decoder)
                    foreground = self.foreground_head(foreground_feature)
                elif self.foreground_arch == 'v3':
                    foreground_feature = torch.cat([semantic_decoder, instance_decoder], dim=1)
                    foreground_feature = self.foreground_fuse(foreground_feature)
                    foreground = self.foreground_head(foreground_feature)
                elif self.foreground_arch == 'v4':
                    foreground_feature = self.foreground_fuse(pred['semantic'])
                    foreground = self.foreground_head(foreground_feature)
                elif self.foreground_arch == 'v5':
                    foreground_feature = torch.cat([pred['center'], pred['offset']], dim=1)
                    foreground_feature = self.foreground_fuse(foreground_feature)
                    foreground = self.foreground_head(foreground_feature)
                elif self.foreground_arch == 'v6':
                    foreground_feature = torch.cat([pred['semantic'], pred['center'], pred['offset']], dim=1)
                    foreground_feature = self.foreground_fuse(foreground_feature)
                    foreground = self.foreground_head(foreground_feature)
                elif self.foreground_arch == 'v7':
                    foreground_feature = torch.cat([semantic_decoder, instance_decoder, pred['semantic']], dim=1)
                    foreground_feature = self.foreground_fuse(foreground_feature)
                    foreground = self.foreground_head(foreground_feature)
                else:
                    raise ValueError('Undefined foreground_arch: {}'.format(self.foreground_arch))
                for key in foreground.keys():
                    pred[key] = foreground[key]

        return pred
