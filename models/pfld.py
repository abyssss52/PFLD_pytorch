#!/usr/bin/env python3
# -*- coding:utf-8 -*-

######################################################
#
# pfld.py -
# written by  zhaozhichao
#
######################################################

import torch
import torch.nn as nn
import math


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=False))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=False))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=False),
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=False),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self, drop_prob):
        super(PFLDInference, self).__init__()
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU6(inplace=False)

        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=False)

        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)

        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)

        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)

        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)  # [16, 14, 14]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, kernel_size=7, stride=1, padding=0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        # self.avg_pool1 = nn.AvgPool2d(14)
        # self.avg_pool2 = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(p=self.drop_prob)

        # self.fc = nn.Linear(4832, 196)    # (176, 196)
        self.fc = nn.Linear(4832, 34)    # (176, 34)

        # self.feature = self.fc

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        # print('S1的维度：', x.shape)
        # x1 = self.avg_pool1(x)          # 方法一
        # print('pool后x1的维度：', x1.shape)
        # x1 = x1.view(x1.size(0), -1)
        x1 = torch.flatten(x, start_dim=1, end_dim=-1)  # 方法二
        # print('x1的维度：', x1.shape)

        x = self.conv7(x)
        # print('S2的维度：', x.shape)
        # x2 = self.avg_pool2(x)         # 方法一
        # print('pool后x2的维度：', x2.shape)
        # x2 = x2.view(x2.size(0), -1)
        x2 = torch.flatten(x, start_dim=1, end_dim=-1)  # 方法二
        # print('x2的维度：', x2.shape)


        x3 = self.relu(self.bn8(self.conv8(x)))
        # print('S3的维度：', x3.shape)
        # x3 = x3.view(x1.size(0), -1)   # 方法一
        x3 = torch.flatten(x3, start_dim=1, end_dim=-1) # 方法二
        # print('x3的维度：', x3.shape)

        multi_scale = torch.cat([x1, x2, x3], 1)
        # print('concat后的维度：', multi_scale.shape)
        # self.feature = multi_scale                            # 可视化需要提取的特征
        multi_scale = self.dropout(multi_scale)
        landmarks = self.fc(multi_scale)

        return out1, landmarks


class AuxiliaryNet(nn.Module):
    def __init__(self, drop_prob):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(64, 128, 3, 2)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.dropout = nn.Dropout(p=drop_prob)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        # self.feature = x                            # 可视化需要提取的特征
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


# if __name__ == '__main__':
#     input = torch.randn(1, 3, 112, 112)
#     plfd_backbone = PFLDInference()
#     auxiliarynet = AuxiliaryNet()
#     features, landmarks = plfd_backbone(input)
#     angle = auxiliarynet(features)

#     print("angle.shape:{0:}, landmarks.shape: {1:}".format(
#         angle.shape, landmarks.shape))
