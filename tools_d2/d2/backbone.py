from torch import nn

from detectron2.layers import BatchNorm2d, NaiveSyncBatchNorm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.utils import env

from segmentation.config.hrnet_config import MODEL_CONFIGS as HRNET_DEFAULT_CONFIGS
from segmentation.model.backbone.hrnet import HighResolutionNet
from segmentation.model.backbone.xception import Xception65


@BACKBONE_REGISTRY.register()
class d2_hrnet(HighResolutionNet, Backbone):
    def __init__(self, cfg, input_shape):
        # fmt: off
        self._out_features  = cfg.MODEL.RESNETS.OUT_FEATURES
        width               = cfg.MODEL.RESNETS.DEPTH  # reuse "DEPTH" for the width in HRNet
        norm                = cfg.MODEL.RESNETS.NORM
        # fmt: on
        assert width in [18, 32, 48]

        self._out_feature_strides = {
            "stem": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "stem": 64,
            "res2": HRNET_DEFAULT_CONFIGS["hrnet%d" % width].STAGE4.NUM_CHANNELS[0],
            "res3": HRNET_DEFAULT_CONFIGS["hrnet%d" % width].STAGE4.NUM_CHANNELS[1],
            "res4": HRNET_DEFAULT_CONFIGS["hrnet%d" % width].STAGE4.NUM_CHANNELS[2],
            "res5": HRNET_DEFAULT_CONFIGS["hrnet%d" % width].STAGE4.NUM_CHANNELS[3],
        }

        norm_layer = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
        super().__init__(HRNET_DEFAULT_CONFIGS["hrnet%d" % width], norm_layer)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"Xception takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
    @property
    def size_divisibility(self):
        return 32


@BACKBONE_REGISTRY.register()
class d2_xception_65(Xception65, Backbone):
    def __init__(self, cfg, input_shape):
        # fmt: off
        self._out_features  = cfg.MODEL.RESNETS.OUT_FEATURES
        res4_dilation       = cfg.MODEL.RESNETS.RES4_DILATION
        res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
        norm                = cfg.MODEL.RESNETS.NORM
        # fmt: on
        assert res4_dilation in {1, 2}, "res4_dilation cannot be {}.".format(res4_dilation)
        assert res5_dilation in {1, 2, 4}, "res5_dilation cannot be {}.".format(res5_dilation)
        if res4_dilation == 2:
            # Always dilate res5 if res4 is dilated.
            assert res5_dilation == 4

        self._out_feature_strides = {
            "stem": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "stem": 64,
            "res2": 256,
            "res3": 728,
            "res4": 728,
            "res5": 2048,
        }

        replace_stride_with_dilation = [False, False, False]
        if res5_dilation > 1:
            replace_stride_with_dilation[-1] = True
            self._out_feature_strides["res5"] = self._out_feature_strides["res5"] // 2
        if res4_dilation > 1:
            replace_stride_with_dilation[-2] = True
            self._out_feature_strides["res4"] = self._out_feature_strides["res4"] // 2
            self._out_feature_strides["res5"] = self._out_feature_strides["res5"] // 2
        norm_layer = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
        super().__init__(replace_stride_with_dilation, norm_layer)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"Xception takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
    @property
    def size_divisibility(self):
        return 32
