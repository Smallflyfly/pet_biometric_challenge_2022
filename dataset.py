#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/5/17 16:43 
"""
import os

import pandas
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DogDataset(Dataset):
    def __init__(self, root_path, label_file, image_list=None, mode='train', width=310):
        self.root_path = root_path
        self.label_file = label_file
        self.image_list = image_list
        self.mode = mode
        self.train_images = []
        self.train_labels = []
        self.label_map = {}

        self.transforms = transforms.Compose([
            transforms.Resize((width, width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.32889438, 0.30539099, 0.29319464], [0.25418268, 0.2446848 , 0.22591035])
        ]) if self.mode == 'train' else transforms.Compose([
            transforms.Resize((width, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.32889438, 0.30539099, 0.29319464], [0.25418268, 0.2446848 , 0.22591035])
        ])

        self._read_label()
        self._process_data()

    def _read_label(self):
        dataframe = pandas.read_csv(os.path.join(self.root_path, self.label_file), 'r', engine='python',
                                    error_bad_lines=False, encoding='ISO-8859-9', delimiter=',')
        labels = dataframe['dog ID']
        filenames = dataframe['nose print image']
        for name, label in zip(filenames, labels):
            self.label_map[name] = int(label)

    def _process_data(self):
        images = []
        if self.image_list is not None:
            images = self.image_list
        else:
            images = os.listdir(os.path.join(self.root_path, 'images'))
        for image in self.image_list:
            label = self.label_map[image]
            image = os.path.join(self.root_path, image)
            self.train_images.append(image)
            self.train_labels.append(label)

    def __getitem__(self, index):
        image = self.train_images[index]
        label = self.train_labels[index]
        im = Image.open(image)
        im = self.transforms(im)
        return im, label

    def __len__(self):
        return len(self.train_images)