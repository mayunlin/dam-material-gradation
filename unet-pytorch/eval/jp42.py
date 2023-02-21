import cv2
import pandas as pd
import math
import torch
import os
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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dir_origin_path = '/home/chase/Boyka/data/stone/seg_vis/unet'
dir_save_path = '/home/chase/Boyka/data/stone/seg_vis/unet/jp'
json_path = '/home/chase/Boyka/data/stone1117/ours/all'
img_names = os.listdir(dir_origin_path)
result_list = []
for img_name in img_names:
    if img_name.lower().endswith('.jpg') and 'JP-1' in img_name:
        print(img_name)
        image_path = os.path.join(dir_origin_path, img_name)
        masks = cv2.imread(image_path)
        binary = masks[:, :, 2]
        binary[binary < 100] = 0
        binary[binary >= 100] = 255
        # cv2.imwrite(os.path.join(dir_save_path, img_name.replace('.jpg', '_vis.jpg')), binary)
        # binary =  cv2.cvtColor(binary,cv2.COLOR_HSV2BGR)

        binary = binary.astype(np.uint8)
        binary = cv2.Canny(binary, 50, 150)
        # ratio = 14.83
        ratio = get_ratio(os.path.join(json_path, img_name.replace('.jpg', '.json')))
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)
        result = []
        mm5 = 0
        mm10 = 0
        mm20 = 0
        mm40 = 0
        mm60 = 0
        mm60_ = 0
        cnt, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for idx in range(len(cnt)):
            try:
                ellipse = cv2.fitEllipse(cnt[idx])
                a, b = ellipse[1]
                a = a / ratio
                b = b / ratio
                d = 1.16 * b * math.pow((1.35 * a / b), 0.5)
                # v = 4 / 3 * math.pi * a * b * d / 2
                area = math.pi * a * b
                v = 0.9 * pow(area * area * area / math.pi, 0.5)
                result.append([idx + 1, d, v, a, b, area])
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
            # cv2.ellipse(masks, ellipse, (255, 255, 255), 2)

        total = mm5 + mm10 + mm20 + mm40 + mm60
        result_list.append([img_name.replace('.jpg', ''), 100 * mm5 / total,
                           100 * mm10 / total,
                            100 * mm20 / total, 100 * mm40 / total,
                            100 * mm60 / total])

        print(result_list)
