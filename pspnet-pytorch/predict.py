
import time

import cv2
import numpy as np
from PIL import Image

from pspnet import PSPNet
import os

from tqdm import tqdm

if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    pspnet = PSPNet()
    ame_classes = ["_background_", "particles", "ball"]
    dir_origin_path = "/home/chase/Boyka/data/stone1117/42/test_data/"
    dir_save_path = "/home/chase/Boyka/data/stone/seg_vis/pspnet/"
    img_names = os.listdir(dir_origin_path)

    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            image = Image.open(image_path)
            r_image = pspnet.detect_image(image)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            r_image.save(os.path.join(dir_save_path, img_name))
