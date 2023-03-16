import pandas as pd
import glob
import numpy as np
import pdb
import os
import pydicom as pyd
from skimage.io import imread,imsave
from skimage.transform import resize
from tqdm import tqdm

LOC='/home/jwg356/datasets/rsna-screening/'
AGE=150
H,W = 512, 512 #256,256

df = pd.read_csv(LOC+'train.csv')
print('Found data with %d columns'%len(df))

tmp = df[df.age < AGE]
tmp[~tmp.BIRADS.isna()].to_csv('data_'+repr(AGE)+'yr.csv',index=False)

### Convert dicom to png
#if not os.path.exists(LOC+'image_dataset'):
#    os.mkdir(LOC+'image_dataset')

### Convert dicom to 256x256 png
if not os.path.exists(LOC+'all_small_dataset'):
    os.mkdir(LOC+'all_small_dataset')


ids = tmp.patient_id.unique()
N = len(ids)
print('Found %d unique patients'%N)
for fIdx in tqdm(range(N)):
    f = repr(ids[fIdx])

    if not os.path.exists(LOC+'all_small_dataset/'+f):
        os.mkdir(LOC+'all_small_dataset/'+f)

#    if not os.path.exists(LOC+'image_dataset/'+f):
#        os.mkdir(LOC+'image_dataset/'+f)


    imgs = sorted(glob.glob(LOC+'train_images/'+f+'/*'))
    
    for img in imgs:

        if 'dcm' in img.split('.')[-1]:
            arr = pyd.dcmread(img).pixel_array.astype(int)
        else:
            arr = imread(img)

        arr = (arr/arr.max()*255).astype(np.uint8)
        arr_small = ((resize(arr*1.0,[H,W])/arr.max())*255).astype(np.uint8)

#        imsave(LOC+'image_dataset/'+f+'/'+img.split('/')[-1].split('.')[0]+'.png',arr)
        imsave(LOC+'all_small_dataset/'+f+'/'+img.split('/')[-1].split('.')[0]+'.png',arr_small)


