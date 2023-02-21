import os
import numpy as np
import pandas
import torch
from models.gcnet import GCNet
from torchvision import transforms
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net=GCNet(input_size=1, output_size=5, num_channels=[8, 6])
net.load_state_dict(torch.load('./models/2022-12-11-1338-58.pth'))
net = net.to(device)
net.eval()
transform = transforms.Compose([
    transforms.ToTensor()
])
jp_dict = {"JP-1": [0.17, 0.11, 0.22, 0.29, 0.21],
           "JP-2": [0.14, 0.15, 0.16, 0.28, 0.27],
           "JP-3": [0.15, 0.09, 0.29, 0.33, 0.14],
           "JP-4": [0.15, 0.19, 0.29, 0.24, 0.13],
           "JP-5": [0.15, 0.18, 0.22, 0.27, 0.18],
           "JP-6": [0.16, 0.19, 0.19, 0.24, 0.22],
           }

if __name__ == '__main__':
    data_path = './dataset/result_csv'
    result = []
    for jp in os.listdir(data_path):
        gt = np.array(jp_dict[jp.replace('.csv', '')])
        data = pandas.read_csv(os.path.join(data_path, jp)).values
        temp = [jp.replace('.csv', '')]
        avg = []
        for idx, sample in enumerate(data):
            sample = sample[~np.isnan(sample)].tolist()
            sample = [np.pad(np.array(sample), (0, 196 - len(sample)), mode='constant')]
            sample = np.float32(np.array(sample))
            sample = transform(sample).to(device)
            outputs = net(sample)
            pre = outputs.cpu().detach().numpy().squeeze()
            avg.append(pre)
        avg = np.array(avg)
        avg = np.average(avg, axis=0)
        mae = np.sum(np.abs(gt * 100 - avg * 100)) / 5
        mse = np.sum((gt * 100 - avg * 100) ** 2) / 5
        rmse = pow(mse, 0.5)
        mape = np.mean(np.abs(gt * 100 - avg * 100) / (gt * 100))
        r2 = 1 - np.sum((gt * 100 - avg * 100) ** 2) / np.sum((gt * 100 - np.mean(gt * 100)) ** 2)
        temp.extend(avg.tolist())
        temp.extend(gt.tolist())
        temp.extend([mae, mse, rmse, mape, r2])
        print(temp)
        result.append(temp)
    res = pd.DataFrame(result)
    res.columns = ['jp', 'pre', 'pre', 'pre', 'pre', 'pre', 'gt', 'gt', 'gt', 'gt', 'gt', 'mae', 'mse', 'rmse', 'mape',
                   'r2']
    res.to_csv('./GCNet_result.csv', index=False)
