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
config_file = '/home/chase/Boyka/stone/mmdetection/stone1116/ours_cas/work-dir/cascade_mask_rcnn_r101_fpn_1x_coco.py'
# 模型路径
checkpoint_file = '/home/chase/Boyka/stone/mmdetection/stone1116/ours_cas/work-dir/epoch_12.pth'
json_path = '/home/chase/Boyka/data/stone1117/ours/all'
dir_origin_path = "/home/chase/Boyka/data/stone1117/3"
dir_save_path = '/home/chase/Boyka/data/stone/ins_vis/ours'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = init_detector(config_file, checkpoint_file, device=device)
img_names = os.listdir(dir_origin_path)
result_list = []
for img_name in tqdm(img_names):
    if img_name.lower().endswith(('.jpg')):

        print(img_name)
        image_path = os.path.join(dir_origin_path, img_name)
        img_file = Image.open(image_path)
        img_file = np.asarray(img_file)
        result = inference_detector(model, img_file)
        masks = np.array(result[1])[0]
        masks = np.array(masks) * 255
        masks = masks.astype(np.uint8)

        ratio = get_ratio(os.path.join(json_path, img_name.replace('.jpg', '.json')))
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)
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
                    # 体积
                    if d < 5:
                        mm5 += v
                    elif d < 10:
                        mm10 += v
                    elif d < 20:
                        mm20 += v
                    elif d < 40:
                        mm40 += v
                    else:
                        mm60 += v
                    # 个数
                    # if d < 5:
                    #     mm5 += 1
                    # elif d < 10:
                    #     mm10 += 1
                    # elif d < 20:
                    #     mm20 += 1
                    # elif d < 40:
                    #     mm40 += 1
                    # else:
                    #     mm60 += 1
                    # 短轴
                    # if d < 5:
                    #     mm5 += a
                    # elif d < 10:
                    #     mm10 += a
                    # elif d < 20:
                    #     mm20 += a
                    # elif d < 40:
                    #     mm40 += a
                    # else:
                    #     mm60 += a
                    # if d < 5:
                    #     mm5 += area
                    # elif d < 10:
                    #     mm10 += area
                    # elif d < 20:
                    #     mm20 += area
                    # elif d < 40:
                    #     mm40 += area
                    # else:
                    #     mm60 += area
                except Exception as e:
                    # print(e)
                    continue
                # print(d, v)
                # cv2.ellipse(mask, ellipse, (0, 0, 255), 2)
                break
            # cv2.imwrite(os.path.join(dir_save_path, img_name.split('.')[0] + str(idx) + '.jpg'), mask)
        total = mm5 + mm10 + mm20 + mm40 + mm60
        result_list.append([img_name.replace('.jpg', ''), int(100 * mm5 / total),
                            int(100 * mm10 / total),
                            int(100 * mm20 / total), int(100 * mm40 / total),
                            int(100 * mm60 / total)])
        print(mm5, mm10, mm20, mm40, mm60)
        data1 = pd.DataFrame(result)
        data1.columns = ['颗粒编号', '粒径', '体积', '长轴', '短轴', '面积']
        # data1.to_csv('结果分析.csv', index=False)
        data1.to_csv(os.path.join(dir_save_path, img_name.replace('.jpg', '结果分析.csv')), index=False)

        # break
res = pd.DataFrame(result_list)
res.columns = ['文件名', '<5mm', '5-10mm', '10-20mm', '20-40mm', '40-60mm']
res.to_csv(os.path.join(dir_save_path, './v_result.csv'), index=False)
