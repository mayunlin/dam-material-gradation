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
    print(ball_ratio)

    return ball_ratio


# 配置文件路径
config_file = './dataset/cascade_mask_rcnn_r101_fpn_1x_coco.py'
# 模型路径
checkpoint_file = './dataset/epoch_12.pth'
#测试图片路径，乒乓球的尺寸是从json里面读出来的
dir_origin_path = "./dataset/data"
#结果保存路径
dir_save_path = './dataset/a'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = init_detector(config_file, checkpoint_file, device=device)
max_length = 0
folders = os.listdir(dir_origin_path)
for folder in folders:
    sub_path = os.path.join(dir_origin_path, folder)
    result_list = []
    for img_name in os.listdir(sub_path):

        if img_name.lower().endswith(('.jpg')) and os.path.exists(
                os.path.join(sub_path, img_name.replace('.jpg', '.json'))):
            print(img_name)
            # continue
            image_path = os.path.join(sub_path, img_name)
            img_file = Image.open(image_path)
            img_file = np.asarray(img_file)
            result = inference_detector(model, img_file)
            masks = np.array(result[1])[0]

            ratio = get_ratio(os.path.join(dir_origin_path,folder, img_name.replace('.jpg', '.json')))
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            result = []
            for idx, mask in enumerate(masks):
                mask = np.array(mask) * 255
                binary = mask.astype(np.uint8)

                # binary = cv2.Canny(mask, 5, 1a5)
                # binary = mask
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
                        result.append(a)
                    except Exception as e:
                        # print(e)
                        continue
                    break
                # cv2.imwrite(os.path.join(dir_save_path, img_name.split('.')[0] + str(idx) + '.jpg'), mask)
            result = np.array(result)
            result = np.sort(result)
            if len(result) > max_length:
                max_length = len(result)
            # print(result)
            result_list.append(result)
            print(max_length)
    res = pd.DataFrame(result_list)
    res.to_csv(os.path.join(dir_save_path, f'{folder}.csv'), index=False)
