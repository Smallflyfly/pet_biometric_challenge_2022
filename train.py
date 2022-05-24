#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/5/17 18:45 
"""
import argparse
import os

import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dataset import DogDataset
from loss import FocalLoss
from model import my_resnet101
from utils import seed_it, load_pretrained_weight, build_optimizer, build_scheduler
import numpy as np
import tensorboardX as tb


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=2, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
args = parser.parse_args()

NUM_CLASSES = 6000
K_FOLD = 8
WIDTH = 448
EPOCH = args.EPOCHS
BATCH_SIZE = args.BATCH

writer = tb.SummaryWriter()


def val_model(model, val_dataloader):
    model.eval()
    softmax = nn.Softmax()
    sum = 0
    for index, data in enumerate(val_dataloader):
        image, label = data
        image = image.cuda()
        out = model(image)
        out = softmax(out).cpu().detach().numpy()
        id = np.argmax(out)
        sum += 1 if id == label.numpy()[0] else 0
    return sum / len(val_dataloader)


def run_train(model, train_dataloader, val_dataloader, loss_func, optimizer, scheduler, fold):
    best_acc = 0
    for epoch in range(1, EPOCH + 1):
        model.train()
        for index, data in enumerate(train_dataloader):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out = model(image)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()

            if index % 50 == 0:
                print('Fold:{} Epoch:[{}/{} {}/{}] lr:{:6f} loss:{:6f}:'.format(
                    fold + 1, epoch, EPOCH, index, len(train_dataloader), optimizer.param_groups[-1]['lr'], loss.item()))

            if index % 20 == 0:
                writer.add_scalar('loss', loss, index)
                writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], index)

        scheduler.step()
        val_acc = val_model(model, val_dataloader)
        print('Fold:{}/Epoch:{} val acc: {:6f}'.format(fold + 1, epoch, val_acc))
        writer.add_scalar('val_acc', val_acc, fold * EPOCH + epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_k_train_{}_fold.pth'.format(fold + 1))


def train():
    seed = 2022
    seed_it(seed)
    images = os.listdir(os.path.join('data/train', 'images'))
    images = np.array(images)
    folds = KFold(n_splits=K_FOLD, shuffle=True, random_state=seed).split(range(len(images)))

    model = my_resnet101(num_classes=NUM_CLASSES)
    model_path = 'weights/resnet101-5d3b4d8f.pth'
    model = model.cuda()
    load_pretrained_weight(model, model_path)
    loss_func = FocalLoss(class_num=6000)
    optimizer = build_optimizer(model, optim='adam', lr=0.0005)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine')
    cudnn.benchmark = True

    for fold, (train_idx, val_idx) in enumerate(folds):
        train_dataset = DogDataset('data/train', 'train_data.csv', images[train_idx], width=WIDTH)
        val_dataset = DogDataset('data/train', 'train_data.csv', images[val_idx], width=WIDTH)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
        run_train(model, train_dataloader, val_dataloader, loss_func, optimizer, scheduler, fold)


if __name__ == '__main__':
    train()
    writer.close()