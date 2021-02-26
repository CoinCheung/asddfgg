
import torch
import torch.nn as nn
import torch.nn.functional as F

from .efficientnet_refactor import EfficientNetBackbone
#  from models.efficientnet import EfficientNetBackbone


from .conv_ops import Conv2dWS as Conv2d
from .conv_ops import SphereConv2d


class ConvBlock(nn.Module):

    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.act(feat)
        return feat


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat


class OutputModule(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=None):
        super(OutputModule, self).__init__()
        self.conv = ConvBlock(in_chan, mid_chan, 3, 1, 1)
        self.drop = nn.Dropout(0.1)
        if not up_factor is None:
            n_classes = n_classes * up_factor * up_factor
        #  self.conv_out = nn.Conv2d(mid_chan,
        self.conv_out = SphereConv2d(mid_chan,
        #  self.conv_out = OSLConv2d(mid_chan,
        #  self.conv_out = NormConv2d(mid_chan,
                                  n_classes,
                                  kernel_size=1,
                                  bias=True)
        if not up_factor is None:
            self.up_sample = nn.PixelShuffle(up_factor)
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        logits = self.conv_out(feat)
        #  logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)
        if not self.up_sample is None:
            logits = self.up_sample(logits)
        return logits

    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1.,
                    mode='fan_in', nonlinearity='leaky_relu')
                if not layer.bias is None: nn.init.constant_(layer.bias, 0)

    def get_params(self):
        wd_params, non_wd_params = [], []
        for name, param in self.named_parameters():
            param_len = len(param.size())
            if param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
            elif param.dim() == 1:
                non_wd_params.append(param)
            else:
                print(name)
        return wd_params, non_wd_params


class PyramidModule(nn.Module):

    def __init__(self, n_chan32=320, n_chan16=112, n_chan8=40, mid_chan=256):
        #TODO: see conv should be 311 or 110
        super(PyramidModule, self).__init__()
        self.conv_gap = ConvBlock(n_chan32, mid_chan, 1, 1, 0)
        self.conv32_inner = ConvBlock(n_chan32, mid_chan, 3, 1, 1)
        self.conv32_outer = ConvBlock(mid_chan, mid_chan, 3, 1, 1)

        self.upsample32 = UpSample(mid_chan, 2)
        self.conv16_inner = ConvBlock(n_chan16, mid_chan, 3, 1, 1)
        self.conv16_outer = ConvBlock(mid_chan, mid_chan, 3, 1, 1)

        self.upsample16 = UpSample(mid_chan, 2)
        self.conv8_inner = ConvBlock(n_chan8, mid_chan, 3, 1, 1)
        self.conv8_outer = ConvBlock(mid_chan, mid_chan, 3, 1, 1)


    def forward(self, feat32, feat16, feat8):
        feat_gap = torch.mean(feat32, dim=(2, 3), keepdims=True)
        feat_gap = self.conv_gap(feat_gap)
        feat32_inner = self.conv32_inner(feat32)
        feat32_inner = feat32_inner + feat_gap
        feat32_outer = self.conv32_outer(feat32_inner) # 32x

        feat32_up = self.upsample32(feat32_inner)
        feat16_inner = self.conv16_inner(feat16)
        feat16_inner = feat16_inner + feat32_up
        feat16_outer = self.conv16_outer(feat16_inner) # 16x

        feat8_up = self.upsample16(feat16_outer)
        feat8_inner = self.conv8_inner(feat8)
        feat8_inner = feat8_inner + feat8_up
        feat8_outer = self.conv8_outer(feat8_inner) # 8x

        return feat8_outer, feat16_outer, feat32_outer


class UShapeEffNetB0Backbone(nn.Module):

    def __init__(self, mid_chan=256):
        super(UShapeEffNetB0Backbone, self).__init__()
        self.backbone = EfficientNetBackbone(model_type='b0')
        self.pyramid = PyramidModule(mid_chan=mid_chan)
        self.conv32 = ConvBlock(256, 1024, 3, 1, 1)
        self.conv16 = ConvBlock(256, 1024, 3, 1, 1)
        self.conv8 = ConvBlock(256, 1024, 3, 1, 1)
        #  self.pyramid = PyramidModule(1280, 448, 160, mid_chan=mid_chan)

    def forward(self, x):
        feat4, feat8, feat16, feat32 = self.backbone(x)
        feat8, feat16, feat32 = self.pyramid(feat32, feat16, feat8)
        feat32 = self.conv32(feat32)
        feat16 = self.conv16(feat16)
        feat8 = self.conv8(feat8)

        return feat8, feat16, feat32


class UShapeEffNetB0(nn.Module):

    def __init__(self, n_classes=19):
        super(UShapeEffNetB0, self).__init__()
        self.backbone = UShapeEffNetB0Backbone(mid_chan=256)
        #  self.out = OutputModule(256, 128, n_classes, up_factor=8)
        self.out = OutputModule(1024, 128, n_classes, up_factor=8)

    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)
        logits = self.out(feat8)

        return logits, feat16, feat32


class UShapeEffNetB0TrainWrapper(nn.Module):

    def __init__(self, n_classes=19):
        super(UShapeEffNetB0TrainWrapper, self).__init__()
        self.model = UShapeEffNetB0(n_classes)
        self.out16 = OutputModule(256, 64, n_classes, up_factor=16)
        self.out32 = OutputModule(256, 64, n_classes, up_factor=32)
        self.n_aux_heads = 2

    def forward(self, x):
        logits, feat16, feat32 = self.model(x)
        logits16 = self.out16(feat16)
        logits32 = self.out32(feat32)

        return logits, logits16, logits32


class UShapeEffNetB0ClassificationWrapper(nn.Module):

    def __init__(self, n_classes=1000):
        super(UShapeEffNetB0ClassificationWrapper, self).__init__()
        self.backbone = UShapeEffNetB0Backbone(mid_chan=256)
        self.classifier = nn.Linear(256 * 4, n_classes)
        self.n_aux_heads = 2

    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)
        tsize = feat16.size()[2:]
        feat = F.interpolate(feat32, size=tsize, mode='nearest') + feat16

        tsize = feat8.size()[2:]
        feat = F.interpolate(feat, size=tsize, mode='nearest') + feat8

        feat = torch.mean(feat, dim=(2, 3))
        logits = self.classifier(feat)

        return logits


    def get_states(self):
        state = dict(
            backbone=self.backbone.state_dict(),
            classifier=self.classifier.state_dict())
        return state


if __name__ == "__main__":
    inten = torch.randn(8, 3, 256, 256)
    net = UShapeEfficientNetB0(18)
    net.eval()
    logits, feat16, feat32 = net(inten)
    print(logits.size())
    print(feat16.size())
    print(feat32.size())

    net = UShapeEfficientNetB0TrainWrapper(18)
    net.eval()
    logits, feat16, feat32 = net(inten)
    print(logits.size())
    print(feat16.size())
    print(feat32.size())
