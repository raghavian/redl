import torch
import glob
import pandas as pd
import numpy as np
from skimage.io import imread
import pdb
from tqdm import tqdm

LOC='/home/jwg356/datasets/rsna-screening/'

subjects = sorted(glob.glob(LOC+'small_dataset/*'))#[:8]
N = len(subjects)
print('Found %d folders'%N)

df = pd.read_csv('/home/jwg356/playground/rsna-screening/data/data_150yr.csv')
ids = df.patient_id.unique()

data = []
labels = []
skip = 0
for sIdx in tqdm(range(N)):
    sub = subjects[sIdx]

    imgs = sorted(glob.glob(sub+'/*.png'))
    for i in imgs:
#        pdb.set_trace()
        idx = df.image_id == int(i.split('/')[-1].split('.')[0]) 
        if len(idx[idx == True]) > 0:

            label = df[idx].values[0]
            img = imread(i)
            data.append(torch.ByteTensor(img))
            labels.append(label)
        else:
#            pdb.set_trace()
            skip += 1
print('Skipped %d images'%skip)
print(len(data),len(labels))
torch.save((data,labels),'rsna_256x256.pt')
