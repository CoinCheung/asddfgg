import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d





SCALING_MAP = {
    '49s': {
        'endpoints_num_filters': 128,
        'filter_size_scale': 0.65,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '49': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '96': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 2,
    },
    '143': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 1.0,
        'block_repeats': 3,
    },
    '190': {
        'endpoints_num_filters': 512,
        'filter_size_scale': 1.3,
        'resample_alpha': 1.0,
        'block_repeats': 4,
    },
}


class ConvBlock(nn.Module):

    def __init__(self,
                 in_chan,
                 out_chan,
                 ks=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_chan,
                           out_chan,
                           kernel_size=ks,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=groups,
                           bias=bias)
        self.norm = nn.BatchNorm2d(out_chan)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

    def fuse_conv_bn(self):
        self.conv = torch.nn.utils.fuse_conv_bn_eval(self.conv, self.norm)
        self.norm = nn.Identity()


class BasicBlock(nn.Module):

    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = nn.Conv2d(
            out_chan,
            out_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_chan,
                    out_chan,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                BatchNorm2d(out_chan),
            )
        self.init_weight()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out

    def init_weight(self):
        for _, md in self.named_modules():
            if isinstance(md, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    md.weight, a=0, mode='fan_out', nonlinearity='leaky_relu'
                )
                if not md.bias is None: nn.init.constant_(md.bias, 0)
        nn.init.constant_(self.bn2.weight, 0)  # gamma of last bn in residual path is initialized to be 0




class Bottleneck(nn.Module):

    def __init__(self,
                 in_chan,
                 out_chan,
                 stride=1,
                 stride_at_1x1=False,
                 dilation=1,
                 use_se=False):
        super(Bottleneck, self).__init__()

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)
        assert out_chan % 4 == 0
        mid_chan = out_chan // 4

        self.conv1 = Conv2d(in_chan,
                            mid_chan,
                            kernel_size=1,
                            stride=stride1x1,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.conv2 = Conv2d(mid_chan,
                            mid_chan,
                            kernel_size=3,
                            stride=stride3x3,
                            padding=dilation,
                            dilation=dilation,
                            bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.conv3 = Conv2d(mid_chan,
                            out_chan,
                            kernel_size=1,
                            bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        #  self.bn3.last_bn = True
        self.relu = nn.ReLU(inplace=True)

        self.use_se = use_se
        if use_se:
            self.se_att = SEBlock(out_chan, 16)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan)
            )
        self.init_weight()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = F.relu(residual, inplace=True)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = F.relu(residual, inplace=True)
        residual = self.conv3(residual)
        residual = self.bn3(residual)

        if self.use_se:
            residual = self.se_att(residual)

        inten = x
        if not self.downsample is None:
            inten = self.downsample(x)
        out = residual + inten
        out = self.relu(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, (Conv2d, nn.Conv2d)):
                nn.init.kaiming_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
        nn.init.constant_(self.bn3.weight, 0)  # gamma of last bn in residual path is initialized to be 0


class BlockSpec(object):
  """A container class that specifies the block configuration for SpineNet."""

  def __init__(self, level, block_fn, input_offsets, is_output):
    self.level = level
    self.block_fn = block_fn
    self.input_offsets = input_offsets
    self.is_output = is_output


FILTER_SIZE_MAP = {
    1: 32,
    2: 64,
    3: 128,
    4: 256,
    5: 256,
    6: 256,
    7: 256,
}

# The fixed SpineNet architecture discovered by NAS.
# Each element represents a specification of a building block:
#   (block_level, block_fn, (input_offset0, input_offset1), is_output).
SPINENET_BLOCK_SPECS = [
    (2, Bottleneck, (None, None), False),  # init block
    (2, Bottleneck, (None, None), False),  # init block
    (2, Bottleneck, (0, 1), False),
    (4, BasicBlock, (0, 1), False),
    (3, Bottleneck, (2, 3), False),
    (4, Bottleneck, (2, 4), False),
    (6, BasicBlock, (3, 5), False),
    (4, Bottleneck, (3, 5), False),
    (5, BasicBlock, (6, 7), False),
    (7, BasicBlock, (6, 8), False),
    (5, Bottleneck, (8, 9), False),
    (5, Bottleneck, (8, 10), False),
    (4, Bottleneck, (5, 10), True),
    (3, Bottleneck, (4, 10), True),
    (5, Bottleneck, (7, 12), True),
    (7, Bottleneck, (5, 14), True),
    (6, Bottleneck, (12, 14), True),
]


def build_block_specs(block_specs=None):
  """Builds the list of BlockSpec objects for SpineNet."""
  if not block_specs:
    block_specs = SPINENET_BLOCK_SPECS
  return [BlockSpec(*b) for b in block_specs]


class Resample(nn.Module):

    def __init__(self, in_channels, out_channels, scale, block_type, alpha=1.0):
        super(Resample, self).__init__()
        self.scale = scale
        new_in_channels = int(in_channels * alpha)
        if block_type == Bottleneck:
            in_channels *= 4
        self.squeeze_conv = ConvBlock(in_channels, new_in_channels, 1, 1, 0)
        if scale < 1:
            self.downsample_conv = ConvBlock(new_in_channels, new_in_channels, 3, 2, 1)
        self.expand_conv = nn.Sequential(
                Conv2d(new_in_channels, out_channels, 1, 1, 0),
                nn.BatchNorm2d(out_channels))


    def _resize(self, x, target_size):

        if self.scale == 1:
            return x
        elif self.scale > 1:
            #  return F.interpolate(x, scale_factor=self.scale, mode='nearest')
            return F.interpolate(x, size=target_size, mode='nearest')
        else:
            x = self.downsample_conv(x)
            if self.scale < 0.5:
                new_kernel_size = 3 if self.scale >= 0.25 else 5
                x = F.max_pool2d(x, kernel_size=new_kernel_size, stride=int(0.5/self.scale), padding=new_kernel_size//2)
            return x

    def forward(self, inputs, target_size):
        feat = self.squeeze_conv(inputs)
        feat = self._resize(feat, target_size)
        feat = self.expand_conv(feat)
        return feat


class Merge(nn.Module):
    """Merge two input tensors"""
    def __init__(self, block_spec, alpha, filter_size_scale):
        super(Merge, self).__init__()
        out_channels = int(FILTER_SIZE_MAP[block_spec.level] * filter_size_scale)
        if block_spec.block_fn == Bottleneck:
            out_channels *= 4

        input0 = block_spec.input_offsets[0]
        input1 = block_spec.input_offsets[1]
        spec0 = BlockSpec(*SPINENET_BLOCK_SPECS[input0])
        spec1 = BlockSpec(*SPINENET_BLOCK_SPECS[input1])
        in_chan0 = int(FILTER_SIZE_MAP[spec0.level] * filter_size_scale)
        in_chan1 = int(FILTER_SIZE_MAP[spec1.level] * filter_size_scale)
        scale0 = 2**(spec0.level - block_spec.level)
        scale1 = 2**(spec1.level - block_spec.level)
        self.resample_op0 = Resample(in_chan0, out_channels, scale0,
                spec0.block_fn, alpha)
        self.resample_op1 = Resample(in_chan1, out_channels, scale1,
                spec1.block_fn, alpha)
        self.scale0, self.scale1 = scale0, scale1

    def compute_out_size(self, feats, scale0, scale1):
        _, _, h0, w0 = feats[0].size()
        _, _, h1, w1 = feats[1].size()
        oh0, ow0 = math.ceil(h0 * scale0), math.ceil(w0 * scale0)
        oh1, ow1 = math.ceil(h1 * scale1), math.ceil(w1 * scale1)
        tsize0, tsize1 = None, None
        if scale0 > 1:
            tsize0 = oh1, ow1
        elif scale1 > 1:
            tsize1 = oh0, ow0
        return tsize0, tsize1

    def forward(self, inputs):
        tsize0, tsize1 = self.compute_out_size(inputs, self.scale0, self.scale1)
        parent0_feat = self.resample_op0(inputs[0], tsize0)
        parent1_feat = self.resample_op1(inputs[1], tsize1)
        target_feat = parent0_feat + parent1_feat
        return target_feat


class SpineNet(nn.Module):
    """Class to build SpineNet backbone"""
    def __init__(self,
                 model_type='49s', in_chan=3, output_level=[3, 4, 5, 6, 7],):
        super(SpineNet, self).__init__()
        self._block_specs = build_block_specs()[2:]
        self._endpoints_num_filters = SCALING_MAP[model_type]['endpoints_num_filters']
        self._resample_alpha = SCALING_MAP[model_type]['resample_alpha']
        self._block_repeats = SCALING_MAP[model_type]['block_repeats']
        self._filter_size_scale = SCALING_MAP[model_type]['filter_size_scale']
        self._init_block_fn = Bottleneck
        self._num_init_blocks = 2
        assert min(output_level) > 2 and max(output_level) < 8, "Output level out of range"
        self.output_level = output_level

        self._make_stem_layer(in_chan)
        self._make_scale_permuted_network()
        self._make_endpoints()
        self.out_chans = [self._endpoints_num_filters for el in range(4)]


    def create_stage(self, block, in_chan, out_chan, b_num, stride=1):
        if block == Bottleneck:
            assert out_chan % 4 == 0
        blocks = [block(in_chan, out_chan, stride=stride),]
        for i in range(1, b_num):
            blocks.append(block(out_chan, out_chan, stride=1))
        return nn.Sequential(*blocks)


    def _make_stem_layer(self, in_chan):
        """Build the stem network."""
        # Build the first conv and maxpooling layers.
        self.conv1 = ConvBlock(in_chan, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build the initial level 2 blocks.
        self.init_block1 = self.create_stage(
            Bottleneck, 64,
            int(FILTER_SIZE_MAP[2] * self._filter_size_scale) * 4,
            self._block_repeats, stride=1
        )
        self.init_block2 = self.create_stage(
            Bottleneck,
            int(FILTER_SIZE_MAP[2] * self._filter_size_scale) * 4,
            int(FILTER_SIZE_MAP[2] * self._filter_size_scale) * 4,
            self._block_repeats, stride=1
        )


    def _make_endpoints(self):
        self.endpoint_convs = nn.ModuleDict()
        for block_spec in self._block_specs:
            if block_spec.is_output:
                in_channels = int(FILTER_SIZE_MAP[block_spec.level]*self._filter_size_scale) * 4
                self.endpoint_convs[str(block_spec.level)] = ConvBlock(
                        in_channels,
                        self._endpoints_num_filters,
                        1, 1, 0)


    def _make_scale_permuted_network(self):
        self.merge_ops = nn.ModuleList()
        self.scale_permuted_blocks = nn.ModuleList()
        for spec in self._block_specs:
            self.merge_ops.append(
                Merge(spec, self._resample_alpha, self._filter_size_scale)
            )
            channels = int(FILTER_SIZE_MAP[spec.level] * self._filter_size_scale)
            in_channels = channels * 4 if spec.block_fn == Bottleneck else channels
            out_chan = channels * 4 if spec.block_fn == Bottleneck else channels

            self.scale_permuted_blocks.append(
                self.create_stage(spec.block_fn,
                               in_channels,
                               out_chan,
                               self._block_repeats, ## 1 for 49, 49s, 2 for 96, 3 for 143
                               stride=1
                               )
            )


    def forward(self, input):
        feat = self.maxpool(self.conv1(input))
        feat1 = self.init_block1(feat)
        feat2 = self.init_block2(feat1)
        block_feats = [feat1, feat2]
        output_feat = {}
        num_outgoing_connections = [0, 0]

        for i, spec in enumerate(self._block_specs):
            target_feat = self.merge_ops[i]([block_feats[feat_idx] for feat_idx in spec.input_offsets])
            # Connect intermediate blocks with outdegree 0 to the output block.
            if spec.is_output:
                for j, (j_feat, j_connections) in enumerate(
                        zip(block_feats, num_outgoing_connections)):
                    if j_connections == 0 and j_feat.shape == target_feat.shape:
                        target_feat += j_feat
                        num_outgoing_connections[j] += 1
            target_feat = F.relu(target_feat, inplace=True)
            target_feat = self.scale_permuted_blocks[i](target_feat)
            block_feats.append(target_feat)
            num_outgoing_connections.append(0)
            for feat_idx in spec.input_offsets:
                num_outgoing_connections[feat_idx] += 1
            if spec.is_output:
                output_feat[spec.level] = target_feat

        return [feat2, ] + [self.endpoint_convs[str(level)](output_feat[level]) for level in self.output_level]


class SpineNetClassificationWrapper(nn.Module):

    def __init__(self, model_type='49s', n_classes=1000):
        super(SpineNetClassificationWrapper, self).__init__()
        self.backbone = SpineNet(model_type)
        mid_chan = SCALING_MAP[model_type]['endpoints_num_filters']
        self.classifier = nn.Linear(mid_chan, n_classes)


    def forward(self, x):
        feat4, feat8, feat16, feat32, feat64, feat128 = self.backbone(x)
        tsize = feat64.size()[2:]
        feat = F.interpolate(feat128, size=tsize, mode='nearest') + feat64

        tsize = feat32.size()[2:]
        feat = F.interpolate(feat, size=tsize, mode='nearest') + feat32

        tsize = feat16.size()[2:]
        feat = F.interpolate(feat, size=tsize, mode='nearest') + feat16

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



if __name__ == '__main__':
    model = SpineNet('49', output_level=[3, 4, 5])
    inten = torch.randn(2, 3, 224, 224)
    outs = model(inten)
    for out in outs:
        print(out.size())
    #  torch.save(model.state_dict(), 'raw.pth')
    #
    #  model = SpineNetClassificationWrapper('49s', 1000)
    #  inten = torch.randn(2, 3, 224, 224)
    #  logits = model(inten)
    #  print(logits.size())
