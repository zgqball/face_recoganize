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


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../../pretrained_model/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--featured_mat', default='', type=str, help='did calculate features')
parser.add_argument('--test_data_mtcnn', default=0, type=int, help='do test data needs transfrom')
args = parser.parse_args()
# print(args)



if args.test_data_mtcnn == 0:
    jpgs_path = r'../../test_mask_datapics/test_data/'
else:
    jpgs_path = r'../../test_mtcnn_data/'

sample_sub = open('../../raw_data/submission_template.csv', 'r')  # sample submission file dir
model_name = args.model.split('/')[4]
epoch_num = args.model.split(',')[1]
sub = open('submission_'+  model_name + '_' + epoch_num + '_with_mask.csv', 'w')
print('Loaded CSV')

name_list = [name for name in os.listdir(jpgs_path)]
img_paths = [jpgs_path + name for name in os.listdir(jpgs_path)]
print('Images number:', len(img_paths))
now_datatime_str = datetime.now().strftime("%Y%m%d")
if args.featured_mat == '':
    model = face_model.FaceModel(args)
    s = time.time()
    if args.test_data_mtcnn == 0:
        features = get_featurs_new(model, img_paths)
        mat_path = 'face_embedding_'+model_name+'_'+now_datatime_str+'_no_mtcnn.mat'
    else:
        features = get_featurs(model, img_paths)
        mat_path = 'face_embedding_'+model_name+'_'+now_datatime_str+'_yes_mtcnn.mat'
    t = time.time() - s
    print(features.shape)
    print('total time is {}, average time is {}'.format(t, t / len(img_paths)))

    fe_dict = get_feature_dict(name_list, features)
    print('Output number:', len(fe_dict))

    
    sio.savemat(mat_path, fe_dict)


    face_features = sio.loadmat(mat_path)
else:
    face_features = sio.loadmat(args.featured_mat)
    
######## cal_submission.py #########
    
lines = sample_sub.readlines()
pbar = tqdm(total=len(lines))
for line in lines:
    pair = line.split(',')[0]
    sub.write(pair + ',')
    a, b = pair.split(':')
    score = '%.5f' % (0.5 + 0.5 * (cosin_metric(face_features[a][0], face_features[b][0])))
    sub.write(score + '\n')
    pbar.update(1)

sample_sub.close()
sub.close()
