import face_model
import argparse
import cv2
import sys
import numpy as np
from tqdm import tqdm
from dataset import Dataset
import torch
from torch.utils import data
import os
import time
import scipy.io as sio
from datetime import datetime
from easydict import EasyDict as edict
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

def get_featurs_new(model, test_list):
    pbar = tqdm(total=len(test_list))
    for idx, img_path in enumerate(test_list):
        pbar.update(1)
        if idx == 0:
            
            img = cv2.imread(img_path)
            new_image = np.transpose(img, (1, 2, 0))[:, :, ::-1]
            new_image = resize_image(img , 112 , 112)
            aligned = np.transpose(new_image, (2,0,1))
            feature = model.get_feature(aligned)
            features = [feature]
        else:
            img = cv2.imread(img_path)
            new_image = np.transpose(img, (1, 2, 0))[:, :, ::-1]
            new_image = resize_image(img , 112 , 112)
            aligned = np.transpose(new_image, (2,0,1))
            feature = model.get_feature(aligned)
            features = np.append(features, [feature], axis=0)
            
    return features



def get_featurs(model, test_list):
    pbar = tqdm(total=len(test_list))
    for idx, img_path in enumerate(test_list):
        pbar.update(1)
        if idx == 0:
            
            img = cv2.imread(img_path)
#             img = model.get_input(img)
            aligned = np.transpose(img, (2,0,1))
            feature = model.get_feature(aligned)
#             feature = feature.detach().cpu().numpy()
            features = [feature]
        else:
#             print(img_path)
            img = cv2.imread(img_path)
#             img = model.get_input(img)
            aligned = np.transpose(img, (2,0,1))
            feature = model.get_feature(aligned)
#             feature = feature.detach().cpu().numpy()
            features = np.append(features, [feature], axis=0)
            
    return features

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    return fe_dict

args = edict()
# general
args.image_size = '112,112'
args.model = ''
args.ga_model = ''
args.gpu = 0
args.det = 0
args.flip = 0
args.threshold = 1.24
args.featured_mat = ''
args.test_data_mtcnn = 0

if args.test_data_mtcnn == 0:
    jpgs_path = r'../../test_data/'
else:
    jpgs_path = r'../../test_mtcnn_data/'

sample_sub = open('../../raw_data/submission_template.csv', 'r')  # sample submission file dir

class_list = ['Indian','Caucasian','Asian','African']
sub = open('submission_diff_clss.csv', 'w')

name_list = [name for name in os.listdir(jpgs_path)]
img_paths = [jpgs_path + name for name in os.listdir(jpgs_path)]

count_class_pic = {'Indian':[],
                'Caucasian':[],
                'Asian':[],
                'African':[],
              }

count_class_pic_path = {'Indian':[],
                'Caucasian':[],
                'Asian':[],
                'African':[],
              }

diff_class_feature_dict = {'Indian':{},
                'Caucasian':{},
                'Asian':{},
                'African':{},
              }
now_datatime_str = datetime.now().strftime("%Y%m%d")
for name in os.listdir(jpgs_path):
    if name.split('_')[0] == 'Indian':
        count_class_pic['Indian'].append(name)
        count_class_pic_path['Indian'].append(jpgs_path + name)
    if name.split('_')[0] == 'Caucasian':
        count_class_pic['Caucasian'].append(name)
        count_class_pic_path['Caucasian'].append(jpgs_path + name)
    if name.split('_')[0] == 'Asian':
        count_class_pic['Asian'].append(name)
        count_class_pic_path['Asian'].append(jpgs_path + name)
    if name.split('_')[0] == 'African':
        count_class_pic['African'].append(name)
        count_class_pic_path['African'].append(jpgs_path + name)

for class_item in class_list:
    args.model = '../../ccf_data/model/r100-softmax-' + class_item + '/model,1'
    model_name = args.model.split('/')[4]
    epoch_num = args.model.split(',')[1]
    model = face_model.FaceModel(args)
    features = get_featurs_new(model, count_class_pic_path[class_item])
    mat_path = model_name+'_'+epoch_num+'_'+now_datatime_str+'_no_mtcnn.mat'
    fe_dict = get_feature_dict(count_class_pic[class_item], features)
    sio.savemat(mat_path, fe_dict)
    print(f'{model_name} Output number:', len(fe_dict))
    diff_class_feature_dict[class_item] = fe_dict


######## cal_submission.py #########
    
lines = sample_sub.readlines()
pbar = tqdm(total=len(lines))
for line in lines:
    pair = line.split(',')[0]
    sub.write(pair + ',')
    a, b = pair.split(':')
    a_feature = diff_class_feature_dict[a.split('_')[0]][a]
    b_feature = diff_class_feature_dict[b.split('_')[0]][b]
    score = '%.5f' % (0.5 + 0.5 * (cosin_metric(a_feature[0], b_feature[0])))
    sub.write(score + '\n')
    pbar.update(1)

sample_sub.close()
sub.close()
