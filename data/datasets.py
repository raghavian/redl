import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pdb
import glob
import pandas as pd
from numpy import random
from torch.nn.functional import avg_pool2d


class LIDCdataset(Dataset):
     def __init__(self, rater=4, data_file = '/home/image/raghav/datasets/lidc_classification/lidc_class256.pt', 
             transform=None):
          super().__init__()

#          pdb.set_trace()
          self.rater = rater
          self.transform = transform
          self.data, self.targets = torch.load(data_file)
          if rater == 4:
              self.targets = (self.targets.sum(1) > 2).type(torch.FloatTensor)
          else:
              self.targets = self.targets[:,rater].type(torch.FloatTensor)
          self.data = self.data.type(torch.FloatTensor)/255.0

     def __len__(self):
          return len(self.targets)

     def __getitem__(self, index):
          image, label = self.data[index], self.targets[index]
          if self.transform is not None:
               image = self.transform(image)
          return image, label

class RSNAdataset(Dataset):
    """
    Label ids: 0:site_id, 1:patient_id, 2:image_id, 3:laterality, 4:view, 5:age, 6:cancer, 7:biopsy,
    8:invasive, 9:BIRADS, 10:implant, 11:density, 12:machine_id, 13:difficult_negative_case
    """
    def __init__(self, data_file = 'data/rsna_256x256.pt',
                transform=None,downsample=1):

        super().__init__()
        self.transform = transform
        self.data, self.targets = torch.load(data_file)
        self.data = self.data.type(torch.FloatTensor)/255.0
        if downsample > 1:
            self.data = avg_pool2d(self.data,downsample)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
#        image = image.type(torch.FloatTensor)/255.0
        if self.transform is not None:
                image = self.transform(image)
        return image, 1.0*(label[9] > 0)

