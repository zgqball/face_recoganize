import os
import cv2
import numpy as np
import time
import scipy.io as sio
from collections import OrderedDict
from tqdm import tqdm
from models import *
import torch
# from config import Config
from torch.nn import DataParallel
from data import Dataset
from torch.utils import data
from models import resnet101
from utils import parse_args
from scipy.spatial.distance import pdist
import itertools
# import insightface


def load_image(img_path, filp=False):
    image = cv2.imread(img_path, 3)
    image = image[-96:, :, :]
    image = cv2.resize(image, (112, 112))
    if image is None:
        return None
    if filp:
        image = cv2.flip(image, 1, dst=None)
    return image


def get_featurs(model, test_list):
    features_dict = {}
    device = torch.device("cuda")

    pbar = tqdm(total=len(test_list))
    for idx, img_path in enumerate(test_list):
        pbar.update(1)


        dataset = Dataset(root=img_path,
                      phase='test',
                      input_shape=(1, 112, 112))

        trainloader = data.DataLoader(dataset, batch_size=1)
        for img in trainloader:
            img = img.to(device)
            if idx == 0:
                feature = model(img)
                feature = feature.detach().cpu().numpy()
                features = feature
                features_dict[img_path] = feature
            else:
                feature = model(img)
                feature = feature.detach().cpu().numpy()
                features_dict[img_path] = feature
                features = np.concatenate((features, feature), axis=0)
         
    return features , features_dict


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    return fe_dict

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cosine_similarity(x1, x2):
    X = np.vstack([x1, x2])
    d2 = 1 - pdist(X, 'cosine')
    return d2

# 加载训练过得模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

checkpoint = '../BEST_checkpoint.tar'
print('loading model: {}...'.format(checkpoint))
checkpoint = torch.load(checkpoint)
model = checkpoint['model'].module.to(device)

model.eval()

s = time.time()
INPUT_DATA = r'../raw_data/train_data/mtcnn_training/'
rootdir_list = os.listdir(INPUT_DATA)
idsdir_list = []
for region_name in rootdir_list:
    if os.path.isdir(os.path.join(INPUT_DATA, region_name)):
        class_dir_list = os.listdir(os.path.join(INPUT_DATA, region_name))
        i = 0
        for class_name in class_dir_list:
            if os.path.isdir(os.path.join(INPUT_DATA, region_name , class_name)):
                if i < 100:
                    idsdir_list.append(os.path.join(region_name , class_name))
                    i += 1
                else:
                    break
pic_pair_list = []
all_pic_list = []
for ids_dir in idsdir_list:
    pic_root_path = os.path.join(INPUT_DATA, ids_dir)
    each_pic_list= os.listdir(pic_root_path)
    each_pic_list = [pic_root_path +'/'+ item for item in each_pic_list]
    all_pic_list.extend(each_pic_list)
    ls = itertools.combinations(each_pic_list, 2)
    pic_pair_list.extend(ls)

    
features , feature_dict= get_featurs(model, all_pic_list)

sim_sum = 0
i = 0

pbar = tqdm(total=len(pic_pair_list))
for path_pair_tuple in pic_pair_list:
    sim = cosin_metric(feature_dict[path_pair_tuple[0]], feature_dict[path_pair_tuple[1]].T)
    sim = 0.5 + 0.5 *sim
#     print(path_pair_tuple , sim)
    sim_sum += sim
    pbar.update(1)
#     i += 1
#     if i > 10:
#         break

t = time.time() - s
print(sim_sum)
print('total time is {}, average time is {}'.format(t, t / len(pic_pair_list)))
