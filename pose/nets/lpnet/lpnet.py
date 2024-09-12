from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from .ct.context_block import ContextBlock
from collections import OrderedDict

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, bias=False):
        super(DepthwiseConv2D, self).__init__()
        padding = (kernel_size - 1) // 2

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                        stride=stride, groups=in_channels, bias=bias)

    def forward(self, x):
        out = self.depthwise_conv(x)
        return out


class Bottleneck(nn.Module):
    expansion = 1
    USE_GCB = True

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = DepthwiseConv2D(planes, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        if self.USE_GCB:
            self.gcb4 = ContextBlock(planes)
        else:
            self.gcb4 = None

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.gcb4 is not None:
            out = self.gcb4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        extra = cfg.MODEL.EXTRA
        self.WIDTH_MULT = extra.WIDTH_MULT
        self.inplanes = int(64 * self.WIDTH_MULT)
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.use_gcb = extra.USE_GCB

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, int(64 * self.WIDTH_MULT), kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(int(64 * self.WIDTH_MULT), momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 * self.WIDTH_MULT), layers[0])
        self.layer2 = self._make_layer(block, int(128 * self.WIDTH_MULT), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * self.WIDTH_MULT), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * self.WIDTH_MULT), layers[3])

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        if not cfg.MODEL.OUT_TYPE == 'offset':
            factor = 1
        else:
            factor = 3
        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS * factor,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        block.USE_GCB = self.use_gcb

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    # (3, [256, 256, 256], [4, 4, 4])
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    groups=planes,
                    bias=self.deconv_with_bias))
            layers.append(nn.Conv2d(planes, planes, kernel_size=1,
                                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if 'stage4.2.fuse_layers' in name:
                    continue
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


# Bottleneck*3+2
resnet_spec = {
    14: (Bottleneck, [1, 1, 1, 1]),
    26: (Bottleneck, [2, 2, 2, 2]),
    38: (Bottleneck, [3, 3, 3, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}


def get_pose_net(cfg, is_train, **kwargs):
    """
    LPNet使用了nn.LayerNorm层，但TNN不支持，固改为nn.BatchNorm2d
    :param cfg:
    :param is_train:
    :param kwargs:
    :return:
    """
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers, cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    return model
