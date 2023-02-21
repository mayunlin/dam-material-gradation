import time

import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

from hrnet import HRnet_Segmentation

if __name__ == "__main__":
    hrnet = HRnet_Segmentation()
    name_classes = ["_background_", "particles","ball"]
    dir_origin_path = "/home/chase/Boyka/data/stone1117/42/test_data/"
    dir_save_path = "/home/chase/Boyka/data/stone/seg_vis/hrnet/"
    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            image = Image.open(image_path)
            r_image = hrnet.detect_image(image)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            r_image.save(os.path.join(dir_save_path, img_name))
