import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from utils.highDHA_utils import initialize_weights

BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                               momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                               momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, band, num_classes):
        super(HighResolutionNet, self).__init__()

        # stem net for hsi
        self.conv1 = nn.Conv2d(band, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        # stem net for msi
        self.conv_msi = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_msi = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # stem net for sar
        self.conv_sar = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_sar = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        ##
        num_channels = 64
        block = blocks_dict['BOTTLENECK']
        num_blocks = 4
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        self.msi_layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        self.sar_layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        num_channels = [48, 96]
        block = blocks_dict['BASIC']
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        ###########
        self.stage2_num_branches = 2
        self.stage2, pre_stage_channels = self._make_stage(1, 2, [4, 4], [48, 96], 'BASIC', 'SUM', num_channels)

        num_channels = [48, 96, 192]
        block = blocks_dict['BASIC']
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        ######
        self.stage3_num_branches = 3
        self.stage3, pre_stage_channels = self._make_stage(1, 3, [3, 3, 3], [48, 96, 192], 'BASIC', 'SUM', num_channels)

        num_channels = [48, 96, 192, 384]
        block = blocks_dict['BASIC']
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        #######
        self.stage4_num_branches = 4
        self.stage4, pre_stage_channels = self._make_stage(
            1, 4, [1, 1, 1, 1], [48, 96, 192, 384], 'BASIC', 'SUM', num_channels, multi_scale_output=True)

        last_inp_channels = int(np.sum(pre_stage_channels)) * 3
        self.transconv = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.final_conv = nn.Conv2d(64, num_classes, 1, 1)
        # self.last_layer = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=last_inp_channels,
        #         out_channels=256,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1),
        #     nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(
        #         in_channels=256,
        #         out_channels=128,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1),
        #     nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(
        #         in_channels=128,
        #         out_channels=64,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1),
        #     nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(
        #         in_channels=64,
        #         out_channels=num_classes,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0)
        # )

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
        )
        ##
        self.tanh = nn.Tanh()

        # for m in self.children():
        #     init_weights(m, init_type='kaiming')
        # self.init_weights()
        initialize_weights(self)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, num_module, num_branch, num_block, num_channel, blockk, fuse, num_inchannels,
                    multi_scale_output=True):
        num_modules = num_module
        num_branches = num_branch
        num_blocks = num_block
        num_channels = num_channel
        block = blocks_dict[blockk]
        fuse_method = fuse

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, y, z, D, domain):
        _, _, height, width = x.shape

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        y = self.relu(self.bn_msi(self.conv_msi(y)))
        z = self.relu(self.bn_sar(self.conv_sar(z)))

        x = self.layer1(x)
        y = self.msi_layer1(y)
        z = self.sar_layer1(z)

        x_list = []
        x_list_msi = []
        x_list_sar = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
                x_list_msi.append(self.transition1[i](y))
                x_list_sar.append(self.transition1[i](z))
            else:
                x_list.append(x)
                x_list_msi.append(y)
                x_list_sar.append(z)
        y_list = self.stage2(x_list)
        y_list_msi = self.stage2(x_list_msi)
        y_list_sar = self.stage2(x_list_sar)

        x_list = []
        x_list_msi = []
        x_list_sar = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                if i < self.stage3_num_branches - 1:
                    x_list.append(self.transition2[i](y_list[i]))
                    x_list_msi.append(self.transition2[i](y_list_msi[i]))
                    x_list_sar.append(self.transition2[i](y_list_sar[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
                    x_list_msi.append(self.transition2[i](y_list_msi[-1]))
                    x_list_sar.append(self.transition2[i](y_list_sar[-1]))
            else:
                x_list.append(y_list[i])
                x_list_msi.append(y_list_msi[i])
                x_list_sar.append(y_list_sar[i])
        y_list = self.stage3(x_list)
        y_list_msi = self.stage3(x_list_msi)
        y_list_sar = self.stage3(x_list_sar)

        x_list = []
        x_list_msi = []
        x_list_sar = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage4_num_branches - 1:
                    x_list.append(self.transition3[i](y_list[i]))
                    x_list_msi.append(self.transition3[i](y_list_msi[i]))
                    x_list_sar.append(self.transition3[i](y_list_sar[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
                    x_list_msi.append(self.transition3[i](y_list_msi[-1]))
                    x_list_sar.append(self.transition3[i](y_list_sar[-1]))
            else:
                x_list.append(y_list[i])
                x_list_msi.append(y_list_msi[i])
                x_list_sar.append(y_list_sar[i])
        x = self.stage4(x_list)
        y = self.stage4(x_list_msi)
        z = self.stage4(x_list_sar)
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y1 = F.interpolate(y[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y2 = F.interpolate(y[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y3 = F.interpolate(y[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        z1 = F.interpolate(z[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        z2 = F.interpolate(z[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        z3 = F.interpolate(z[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        # x0_avg = (x[0] + y[0] + z[0]) / 3
        # x1_avg = (x1 + y1 + z1) / 3
        # x2_avg = (x2 + y2 + z2) / 3
        # x3_avg = (x3 + y3 + z3) / 3
        # x = torch.cat([x0_avg, x1_avg, x2_avg, x3_avg], 1)
        x = torch.cat([x[0], x1, x2, x3, y[0], y1, y2, y3, z[0], z1, z2, z3], 1)

        if domain == 'source':
            xt = x
        if domain == 'target':
            aa = D[0](x)
            aa = self.tanh(aa)
            aa = torch.abs(aa)
            aa_big = aa.expand(x.size())
            xt = aa_big * x + x

        out = self.last_layer(xt)
        # out = F.interpolate(out, (height, width), mode='bilinear', align_corners=True)
        out = self.transconv(out)
        out = self.final_conv(out)

        return x, out

    def init_weights(self, pretrained='', ):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
