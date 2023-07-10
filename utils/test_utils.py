#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import warnings
import torch
from torch import nn
from torch import optim
import models
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay


class test_utils(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load the datasets
        if args.processing_type == 'O_A':
            from CNN_Datasets.O_A import datasets
            Dataset = getattr(datasets, args.data_name)
        elif args.processing_type == 'R_A':
            from CNN_Datasets.R_A import datasets
            Dataset = getattr(datasets, args.data_name)
        elif args.processing_type == 'R_NA':
            from CNN_Datasets.R_NA import datasets
            Dataset = getattr(datasets, args.data_name)
        else:
            raise Exception("processing type not implement")

        print(Dataset)

        self.datasets  = Dataset(args.data_dir,args.normlizetype).data_preprare(True)

        self.dataloaders = torch.utils.data.DataLoader(self.datasets, batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
        
        # Define the model
        self.model = getattr(models, args.model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Load the checkpoint
        self.start_epoch = 0

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()


    def test(self):
        """
        Testing process
        :return:
        """
        args = self.args
        valid_running_correct = 0

        best_model = torch.load(args.checkpoint_path)
        self.model.load_state_dict(best_model)
        self.model.eval()

        class_names = self.datasets.class_names

        y_pred = []
        y_true = []

        with torch.set_grad_enabled(False):
            for _, (inputs, labels) in enumerate(self.dataloaders):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                pred = self.model(inputs)
                _, preds = torch.max(pred.data, 1)

                y_pred.extend(preds.tolist()) # Save Prediction
                y_true.extend(labels.tolist()) # Save Truth

                # calculate the accuracy
                _, preds = torch.max(pred.data, 1)
                valid_running_correct += (preds == labels).sum().item()
            
        # final accuracy
        final_acc = 100. * (valid_running_correct / len(self.dataloaders.dataset))
        logging.info('Accuracy on the test dataset of is {}'.format(final_acc))

        cf_matrix = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=class_names)
        disp.plot()
        checkpoint_folder = '/'.join(args.checkpoint_path.rsplit('/')[:-1])
        plt.savefig(os.path.join(checkpoint_folder, args.checkpoint_path.rsplit('/')[-1].rsplit('.pth')[0] + '_confusion_matrix.png'))

    










