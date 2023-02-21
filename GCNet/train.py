#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/11/13 16:37
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : train.py
@Project    : BridgeDefectLocation
@Description:
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.dataset import JPDataset
import logging

import torchvision.transforms as transforms
import torch.utils.data as data
import datetime
# from models.conv1d import GCNet
#
from models.gcnet import GCNet
logdir = './logs'
if not os.path.exists(logdir):
    os.makedirs(logdir)
log_file_name = '%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
log_path = os.path.join(logdir, log_file_name + '.log')

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(fmt=formatter)

file_handler = logging.FileHandler(filename=log_path, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt=formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
#
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Resize(256),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
train_set = JPDataset(data_path='./dataset/result_csv',
                      in_transform=transform,
                      out_transform=transform,
                      dataset_type='train')
test_set = JPDataset(data_path='./dataset/result_csv',
                     in_transform=transform,
                     out_transform=transform,
                     dataset_type='test')
batch_size = 4
train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, num_workers=8, shuffle=True)
test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, num_workers=8, shuffle=False)

num_epochs = 1000
# net = GCNet()
net=GCNet(input_size=1, output_size=5, num_channels=[8, 6])

net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
# 日志输出run_loss，run_acc
for epoch in range(num_epochs):
    gt = []
    pre = []
    net.train()
    for i, data in enumerate(train_loader):
        obj, label = data
        obj, label = obj.to(device), label.to(device)
        outputs = net(obj)
        loss = criterion(outputs, label)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        gt.append(label.cpu().detach().numpy().squeeze())
        pre.append(outputs.cpu().detach().numpy().squeeze())
    gt = np.array(gt).reshape(-1, 5) * 100
    pre = np.array(pre).reshape(-1, 5) * 100
    mae = np.sum(np.abs(gt - pre)) / (gt.shape[0] * gt.shape[1])
    mse = np.sum((gt - pre) ** 2) / (gt.shape[0] * gt.shape[1])
    rmse = pow(mse, 0.5)
    mape = np.mean(np.abs(gt * 100 - pre * 100) / (gt * 100))
    r2 = 1 - np.sum((gt - pre) ** 2) / np.sum((gt - np.mean(gt)) ** 2)
    logger.info('epoch %d --- mae: %.2f,mse: %.2f,rmse: %.2f,mape: %.2f,r2: %.2f' % (epoch, mae, mse, rmse, mape, r2))

    # logger.info('train epoch %d --- mae: %d,mse: %d,r2: %d' % (epoch, mae, mse, r2))
    net.eval()
    with torch.no_grad():
        gt = []
        pre = []
        for i, data in enumerate(test_loader):
            obj, label = data
            obj, label = obj.to(device), label.to(device)
            outputs = net(obj)
            gt.append(label.cpu().detach().numpy().squeeze())
            pre.append(outputs.cpu().detach().numpy().squeeze())
        gt = np.array(gt).reshape(-1, 5) * 100
        pre = np.array(pre).reshape(-1, 5) * 100
        mae = np.sum(np.abs(gt - pre)) / (len(gt) * 5)
        mse = np.sum((gt - pre) ** 2) / (len(gt) * 5)
        rmse = pow(mse, 0.5)
        mape = np.mean(np.abs(gt * 100 - pre * 100) / (gt * 100))
        r2 = 1 - np.sum((gt - pre) ** 2) / np.sum((gt - np.mean(gt)) ** 2)
        logger.info(
            'test epoch %d --- mae: %.2f,mse: %.2f,rmse: %.2f,mape: %.2f,r2: %.2f' % (epoch, mae, mse, rmse, mape, r2))
    if not os.path.exists('./models'):
        os.makedirs('./checkpoints')
    torch.save(net.state_dict(), os.path.join('./checkpoints', log_file_name + '.pth'))
