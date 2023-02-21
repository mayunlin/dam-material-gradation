import cv2
from tqdm import tqdm
from torch import device
import pandas as pd
import math
import torch
import os
from PIL import Image
from mmdet.apis import init_detector, show_result_pyplot, inference_detector
import numpy as np
import json

jp_dict = {"JP-1": [0.17, 0.11, 0.22, 0.29, 0.21],
           "JP-2": [0.14, 0.15, 0.16, 0.28, 0.27],
           "JP-3": [0.15, 0.09, 0.29, 0.33, 0.14],
           "JP-4": [0.15, 0.19, 0.29, 0.24, 0.13],
           "JP-5": [0.15, 0.18, 0.22, 0.27, 0.18],
           "JP-6": [0.16, 0.19, 0.19, 0.24, 0.22],
           }


def get_ratio(json_path):
    ball_ratio = 20
    with open(json_path, 'r', ) as fw:
        info_dict = json.load(fw)
        objs = info_dict['shapes']
        for obj in objs:
            if obj['label'] == 'ball':
                points = np.array(obj['points'])
                height = np.max(points[:, 0]) - np.min(points[:, 0])
                width = np.max(points[:, 1]) - np.min(points[:, 1])
                ball_ratio = np.min([height, width]) / 40

    return ball_ratio

# 配置文件路径
config_file = '/home/chase/Boyka/stone/mmdetection/stone1116/origin_al/cas/cascade_mask_rcnn_r101_fpn_1x_coco.py'
# 模型路径
checkpoint_file = '/home/chase/Boyka/stone/mmdetection/stone1116/origin_al/cas/epoch_12.pth'
json_path = '/home/chase/Boyka/data/stone1117/ours/all'
dir_origin_path = "/home/chase/Boyka/stone/GCNet/dataset/data"
dir_save_path = '/home/chase/Boyka/stone/GCNet/dataset/result1_6'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not os.path.exists(dir_save_path):
    os.makedirs(dir_save_path)
model = init_detector(config_file, checkpoint_file, device=device)
final_result = []
for jp in os.listdir(dir_origin_path):
    if not  'JP-1' in jp:
        continue
    gt = np.array(jp_dict[jp]) * 100
    temp = [jp]
    sub_path = os.path.join(dir_origin_path, jp)
    img_names = os.listdir(sub_path)
    result_list = []
    for img_name in img_names:
        if img_name.lower().endswith(('.jpg')) and \
                os.path.exists(os.path.join(sub_path, img_name.replace('.jpg', '.json'))):
            image_path = os.path.join(sub_path, img_name)
            img_file = Image.open(image_path)
            img_file = np.asarray(img_file)
            result = inference_detector(model, img_file)
            masks = np.array(result[1])[0]
            masks = np.array(masks) * 255
            masks = masks.astype(np.uint8)
            ratio = get_ratio(os.path.join(json_path, img_name.replace('.jpg', '.json')))
            result = []
            mm5 = 0
            mm10 = 0
            mm20 = 0
            mm40 = 0
            mm60 = 0
            for idx, mask in enumerate(masks):
                # binary = cv2.Canny(mask, 5, 1a5)
                binary = mask
                cnt, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for i in range(len(cnt)):
                    try:
                        ellipse = cv2.fitEllipse(cnt[i])
                        a, b = ellipse[1]
                        a = a / ratio
                        b = b / ratio
                        d = 1.16 * b * math.pow((1.35 * a / b), 0.5)
                        v = 4 / 3 * math.pi * a * b * d / 2
                        area = math.pi * a * b
                        result.append([idx + 1, d, v, a, b, area])
                        # 短轴
                        if d < 5:
                            mm5 += a
                        elif d < 10:
                            mm10 += a
                        elif d < 20:
                            mm20 += a
                        elif d < 40:
                            mm40 += a
                        else:
                            mm60 += a
                    except Exception as e:
                        # print(e)
                        continue
                    break
            total = mm5 + mm10 + mm20 + mm40 + mm60
            result_list.append([100 * mm5 / total,
                                100 * mm10 / total,
                                100 * mm20 / total, 100 * mm40 / total,
                                100 * mm60 / total])
            # result_list.append([img_name.replace('.jpg', ''), int(100 * mm5 / total),
            #                     int(100 * mm10 / total),
            #                     int(100 * mm20 / total), int(100 * mm40 / total),
            #                     int(100 * mm60 / total)])
    result_list = np.array(result_list)
    avg = np.average(result_list, axis=0)
    mae = np.sum(np.abs(gt - avg)) / 5
    mse = np.sum((gt - avg) ** 2) / 5
    rmse = pow(mse, 0.5)
    mape = np.mean(np.abs(gt * 100 - avg * 100) / (gt * 100))
    r2 = 1 - np.sum((gt - avg) ** 2) / np.sum((gt - np.mean(gt)) ** 2)
    # mae = np.sum(np.abs(result_list * 100 - avg * 100)) / 5
    # mse = np.sum((gt * 100 - avg * 100) ** 2) / 5
    # rmse = pow(mse, 0.5)
    # r2 = 1 - np.sum((gt * 100 - avg * 100) ** 2) / np.sum((gt * 100 - np.mean(gt * 100)) ** 2)
    temp.extend(avg.tolist())
    temp.extend(gt.tolist())
    temp.extend([mae, mse, rmse, mape, r2])
    # print(temp)
    final_result.append(temp)
    print(final_result[-1])
res = pd.DataFrame(final_result)
res.columns = ['jp', 'pre', 'pre', 'pre', 'pre', 'pre', 'gt', 'gt', 'gt', 'gt', 'gt', 'mae', 'mse', 'rmse', 'mape',
               'r2']
res.to_csv(os.path.join(dir_save_path, 'cas_result.csv'), index=False)
