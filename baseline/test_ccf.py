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
# import insightface

IMAGE_SIZE = 112 
def get_featurs(model, test_list):

    device = torch.device("cuda")

    pbar = tqdm(total=len(test_list))
    for idx, img_path in enumerate(test_list):
        pbar.update(1)


        dataset = Dataset(root=img_path,
                      phase='test',
                      input_shape=(1, 112, 96))

        trainloader = data.DataLoader(dataset, batch_size=1)
        for img in trainloader:
            img = img.to(device)
            if idx == 0:
                feature = model(img)
                feature = feature.detach().cpu().numpy()
                features = feature
            else:
                feature = model(img)
                feature = feature.detach().cpu().numpy()
                features = np.concatenate((features, feature), axis=0)
    return features


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

data_dir = '../test_data/'                      # testset dir
name_list = [name for name in os.listdir(data_dir)]
img_paths = [data_dir + name for name in os.listdir(data_dir)]
print('Images number:', len(img_paths))

s = time.time()
features = get_featurs(model, img_paths)
t = time.time() - s
print(features.shape)
print('total time is {}, average time is {}'.format(t, t / len(img_paths)))

fe_dict = get_feature_dict(name_list, features)
print('Output number:', len(fe_dict))
sio.savemat('face_embedding_test.mat', fe_dict)

######## cal_submission.py #########

face_features = sio.loadmat('face_embedding_test.mat')
print('Loaded mat')
sample_sub = open('../raw_data/submission_template.csv', 'r')  # sample submission file dir
sub = open('submission_new.csv', 'w')
print('Loaded CSV')

lines = sample_sub.readlines()
pbar = tqdm(total=len(lines))
for line in lines:
    pair = line.split(',')[0]
    sub.write(pair + ',')
    a, b = pair.split(':')
    score = '%.5f' % (0.5 + 0.5 * (cosin_metric(face_features[a][0], face_features[b][0])))
#     score = '%.5f' % cosin_metric(face_features[a][0], face_features[b][0])
    # score = '%2.f' % cosine_similarity(face_features[a][0], face_features[b][0])
    sub.write(score + '\n')
    pbar.update(1)

sample_sub.close()
sub.close()