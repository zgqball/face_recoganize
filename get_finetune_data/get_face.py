# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'test'))
import face_model

import argparse
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import shutil

import time
from PIL import Image

class Args():
    def __init__(self):
        self.image_size = '112,112'
        self.gpu = 0
        self.model = './models/model-r50-am-lfw/model,0000'
        self.ga_model = './models/gamodel-r50/model,0000'
        self.threshold = 1.24
        self.flip = 0
        self.det = 0
IMAGE_SIZE = 112 

def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    #獲取影象尺寸
    h, w, _ = image.shape

    #對於長寬不相等的圖片，找到最長的一邊
    longest_edge = max(h, w)    
    #計算短邊需要增加多上畫素寬度使其與長邊等長
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    BLACK = [0, 0, 0]
    #給影象增加邊界，是圖片長、寬等長，cv2.BORDER_CONSTANT指定邊界顏色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    #調整影象大小並返回
    return cv2.resize(constant, (height, width))

args = Args()
model = face_model.FaceModel(args)

path = r'../raw_data/train_data/training/'
dst_path = '../raw_data/train_data/mtcnn_training/'
# imgs = os.listdir(path)
cnt = 0

print('finished load model')
for root, dirs, files in os.walk(path):
    for region_dir in dirs:
        class_root = os.path.join(root, region_dir)
        region_class_dirs = os.listdir(class_root)
        for region_class_dir in region_class_dirs:
            image_root_path = os.path.join(class_root, region_class_dir)
            images_list = os.listdir(image_root_path)
            for image in images_list:
                print(os.path.join(image_root_path, image))
                img = cv2.imread(os.path.join(image_root_path, image))
                out = model.get_input(img)  # 3x112x112
                try:
                    print(f'{out.shape}')
                    new_image = np.transpose(out, (1, 2, 0))[:, :, ::-1]
                except:
                    new_image = np.transpose(img, (1, 2, 0))[:, :, ::-1]
                    new_image = resize_image(img , 112 , 112)
                print(f'{new_image.shape}')
                out = Image.fromarray(new_image)
                out = out.resize((112, 112))
                out = np.asarray(out)

            # for point in points:
            #     cv2.circle(out, (point[0], point[1]), 2, (0, 0, 255), -1)
                #     cv2.putText(image, str(num), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                if not os.path.exists(os.path.join(dst_path, region_dir)):
                    os.mkdir(os.path.join(dst_path, region_dir))
                if not os.path.exists(os.path.join(dst_path, region_dir , region_class_dir)):
                    os.mkdir(os.path.join(dst_path, region_dir , region_class_dir))
                cv2.imwrite(os.path.join(dst_path, region_dir, region_class_dir , image[:-4] + '_mtcnn.jpg'), out)
                cnt += 1
            