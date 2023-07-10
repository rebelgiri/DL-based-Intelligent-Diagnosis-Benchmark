import os
import numpy as np
import pandas as pd
from itertools import islice
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
import logging

signal_size = 1024

# Generate Training Dataset and Testing Dataset


def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    datasetname:List of  dataset
    '''
    datasetname = os.listdir(root)  # 0:training, 1:test
    training_data = os.path.join(root, datasetname[0])  # Path of training
    testing_data = os.path.join(root, datasetname[1])  # Path of test

    if False == test:
        class_names = os.listdir(training_data)
    else:
        class_names = os.listdir(testing_data)

    data = []
    lab = []

    for i in range(len(class_names)):
        logging.info("Class {} labeled as : {}".format(class_names[i], i))

        if False == test:
            class_path = os.path.join(training_data, class_names[i])
        else:
            class_path = os.path.join(testing_data, class_names[i])

        files = os.listdir(class_path)
        for j in tqdm(range(len(files))):
            file_path = os.path.join(class_path, files[j])
            data1, lab1 = data_load(file_path, dataname=files[j], label=i)
            data += data1
            lab += lab1

    return [data, lab], class_names


def data_load(filename, dataname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    f = open(filename, "r", encoding='gb18030', errors='ignore')
    fl = []

    for line in islice(f, 29, None):  # Skip the first 29 lines
        line = line.rstrip()
        word = line.split(";", 8)  # Separated by ;
        # Take a vibration signal in the x direction as input
        fl.append(eval(word[1]))

    fl = np.array(fl)
    fl = fl.reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]/10:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab


def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            RandomAddGaussian(),
            RandomScale(),
            RandomStretch(),
            RandomCrop(),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]
# --------------------------------------------------------------------------------------------------------------------


class DIVIBES(object):
    num_classes = 4
    inputchannel = 1

    def __init__(self, data_dir, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):
        list_data, class_names = get_files(self.data_dir, test)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        if test:
            test_dataset = dataset(list_data=data_pd, class_names=class_names,
                                   test=True, transform=data_transforms('val', self.normlizetype))
            return test_dataset
        else:
            train_pd, val_pd = train_test_split(
                data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            train_dataset = dataset(
                list_data=train_pd, class_names=class_names, transform=data_transforms('train', self.normlizetype))
            val_dataset = dataset(
                list_data=val_pd, class_names=class_names, transform=data_transforms('val', self.normlizetype))
            return train_dataset, val_dataset
