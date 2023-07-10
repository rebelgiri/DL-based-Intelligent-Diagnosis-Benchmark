#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from datasets.sequence_aug import *


class dataset(Dataset):

    def __init__(self, list_data, class_names, test=False, transform=None):
        self.test = test
        if self.test:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
            self.class_names = class_names
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label
