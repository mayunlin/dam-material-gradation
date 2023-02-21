#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/10/12 11:02
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : tcn.py
@Project    : BridgeDefectLocation
@Description:
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GCNet(nn.Module):
    def __init__(self, input_size, output_size, num_channels):
        super(GCNet, self).__init__()
        self.gcn = TemporalConvNet(input_size, num_channels)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4, stride=2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.conv1d_2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.conv1d_3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, stride=2)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.conv1d_4 = nn.Conv1d(in_channels=16, out_channels=4, kernel_size=4, stride=2)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)
        self.out = nn.Linear(40, 5)
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x= self.gcn(x)
        x = self.relu1(self.conv1d_1(x))
        x = self.dropout1(x)
        x = self.relu2(self.conv1d_2(x))
        x = self.dropout2(x)
        x = self.relu3(self.conv1d_3(x))
        x = self.dropout3(x)
        x = self.relu4(self.conv1d_4(x))
        x = self.dropout4(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.out(x)
        x = self.softmax(x)
        return x
        # x= self.gcn(x)
        # x = self.conv1d_1(x)
        # x = self.conv1d_2(x)
        # x = self.conv1d_3(x)
        # x = self.conv1d_4(x)
        # x = torch.flatten(x, start_dim=1, end_dim=-1)
        # x = self.out(x)
        # return x
#

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    model = GCNet(input_size=1, output_size=196, num_channels=[8,  1])
    model = model.to(device)
    print(model)
    a = torch.ones([4, 1, 196])
    a = a.to(device)
    out = model(a)
    print(out.shape)
    #
    # model = model.to(device)
    # print(model)
    # a = torch.ones([1, 1, 196])
    # a = a.to(device)
    # out = model(a)
    # print(out.shape)
    # print(out)
