#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/10/13 16:37
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : dats.py
@Project    :
@Description:
"""
import pandas
import torch.utils.data as data
import os
import pandas as pd
import numpy as np

import torchvision.transforms as transforms
import torch


class JPDataset(data.Dataset):
    def __init__(self, data_path, in_transform, out_transform, dataset_type):
        self.jp_dict = {"JP-1": [0.17, 0.11, 0.22, 0.29, 0.21],
                        "JP-2": [0.14, 0.15, 0.16, 0.28, 0.27],
                        "JP-3": [0.15, 0.09, 0.29, 0.33, 0.14],
                        "JP-4": [0.15, 0.19, 0.29, 0.24, 0.13],
                        "JP-5": [0.15, 0.18, 0.22, 0.27, 0.18],
                        "JP-6": [0.16, 0.19, 0.19, 0.24, 0.22],
                        }
        self.data_scale = 196
        self.data_path = data_path
        self.in_transform = in_transform
        self.out_transform = out_transform
        self.dataset_type = dataset_type
        self.in_data, self.out_data = self.__build_dataset__()
        self.length = self.__len__()

    def __build_dataset__(self):
        in_data = []
        out_data = []
        for jp in os.listdir(self.data_path):
            label = self.jp_dict[jp.replace('.csv', '')]
            samples = pandas.read_csv(os.path.join(self.data_path, jp))
            samples = samples.values
            for idx, sample in enumerate(samples):
                if idx < 8 and self.dataset_type == 'train':
                    sample = sample[~np.isnan(sample)].tolist()
                    in_data.append([np.pad(np.array(sample), (0, self.data_scale - len(sample)), mode='constant')])
                    out_data.append([label])
                if idx > 1 and self.dataset_type == 'test':
                    sample = sample[~np.isnan(sample)].tolist()
                    in_data.append([np.pad(np.array(sample), (0, self.data_scale - len(sample)), mode='constant')])
                    out_data.append([label])
        return in_data, out_data

    def __getitem__(self, index):
        in_data = np.float32(np.array(self.in_data[index]))
        out_data = np.float32(np.array(self.out_data[index]))

        if self.in_transform is not None:
            in_data = self.in_transform(in_data)
        if self.out_transform is not None:
            out_data = self.out_transform(out_data)
        return torch.squeeze(in_data, 1), torch.squeeze(out_data)

    def __len__(self):
        return len(self.in_data)


if __name__ == '__main__':
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    train_set = JPDataset(data_path='./result_csv',
                          in_transform=transform,
                          out_transform=transform,
                          dataset_type='train')

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=8, num_workers=8, shuffle=True)

    for i, train_data in enumerate(train_loader):
        obj, tar = train_data
        print(obj.shape)
