from tqdm import tqdm
from torch import device
import torch
import os
from PIL import Image
from mmdet.apis import init_detector, show_result_pyplot, inference_detector
import numpy as np

# 配置文件路径
config_file = '/home/chase/Boyka/stone/mmdetection/stone1116/origin_al/cas/cascade_mask_rcnn_r101_fpn_1x_coco.py'
# 模型路径
checkpoint_file = '/home/chase/Boyka/stone/mmdetection/stone1116/origin_al/cas/epoch_12.pth'
json_path = '/home/chase/Boyka/data/stone1117/ours/all'
dir_origin_path = "/home/chase/Boyka/data/stone1117/3"
dir_save_path = '/home/chase/Boyka/data/stone/ins_vis/cas'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = init_detector(config_file, checkpoint_file, device=device)
img_names = os.listdir(dir_origin_path)
for img_name in tqdm(img_names):
    if img_name.lower().endswith(
            ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
        image_path = os.path.join(dir_origin_path, img_name)
        img_file = Image.open(image_path)
        img_file = np.asarray(img_file)
        result = inference_detector(model, img_file)
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)
        out_file = os.path.join(dir_save_path, img_name.replace('.jpg','_mask.jpg'))
        # out_file = os.path.join(dir_save_path, img_name)
        img_file=np.zeros(shape=img_file.shape)
        show_result_pyplot(model, img_file, result, score_thr=0.5, out_file=out_file)
        # print(result)
# img_file ='bus.jpg'
# savepath='runs/'
# result = inference_detector(model, img_file)
# print(result)
#
# out_file = os.path.join(savepath, 'bus.jpg')
# print(out_file)
# show_result_pyplot(model, img_file, result,score_thr=0.9,out_file=out_file)
