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
                height=np.max(points[:,0])-np.min(points[:,0])
                width = np.max(points[:, 1]) - np.min(points[:, 1])
                ball_ratio=np.min([height,width])/40
    print(ball_ratio)

    return ball_ratio


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dir_origin_path = '/home/chase/Boyka/data/stone/seg_vis/unet'
dir_save_path = '/home/chase/Boyka/data/stone/seg_vis/unet/jp'
json_path = '/home/chase/Boyka/data/stone1117/ours/all'
img_names = os.listdir(dir_origin_path)
result_list = []
for img_name in img_names:
    if img_name.lower().endswith('.jpg'):
        print(img_name)
        image_path = os.path.join(dir_origin_path, img_name)
        masks = cv2.imread(image_path)
        binary=masks[:,:,2]
        binary[binary<100]=0
        binary[binary>=100]=255
        cv2.imwrite(os.path.join(dir_save_path, img_name.replace('.jpg', '_vis.jpg')), binary)
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
                #体积
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
                #个数
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
                #个数
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
            cv2.ellipse(masks, ellipse, (255, 255, 255), 2)

        total = mm5 + mm10 + mm20 + mm40 + mm60
        result_list.append([img_name.replace('.jpg', ''), int(100 * mm5 / total),
                            int(100 * mm10 / total),
                            int(100 * mm20 / total), int(100 * mm40 / total),
                            int(100 * mm60 / total)])

        # result_list.append([img_name.replace('.jpg', ''), int(100 * mm5 / len(result)+0.5),
        #                     int(100 * mm10 / len(result)+0.5),
        #                     int(100 * mm20 / len(result)+0.5), int(100 * mm40 / len(result)+0.5),
        #                     int(100 * mm60 / len(result)+0.5)])
        print(mm5, mm10, mm20, mm40, mm60)
        print(os.path.join(dir_save_path, img_name.replace('.jpg', '_vis.jpg')))
        cv2.imwrite(os.path.join(dir_save_path, img_name.replace('.jpg', '_vis.jpg')), masks)
        data1 = pd.DataFrame(result)
        data1.columns = ['颗粒编号', '粒径', '体积', '长轴', '短轴', '面积']
        # data1.to_csv('结果分析.csv', index=False)
        data1.to_csv(os.path.join(dir_save_path, img_name.replace('.jpg', '结果分析.csv')), index=False)

        # break
res = pd.DataFrame(result_list)
print(result_list)
res.columns = ['文件名', '<5mm', '5-10mm', '10-20mm', '20-40mm', '40-60mm']
res.to_csv('/home/chase/Boyka/data/stone/seg_vis/unet/v_result.csv', index=False)
