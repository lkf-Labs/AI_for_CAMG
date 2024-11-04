import os
import time
import yaml
import cv2
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from SegDataset import ImageDataset
import kornia.augmentation as KAug
from models.unet_model import UNet
from predict import save_result, evaluate



def getMinLoss(lines):
    min_loss = float('inf')
    for line in lines:
        args = line.split(',')
        if float(args[1]) < float(min_loss):
            min_loss = float(args[1])
    return min_loss

def train(train_range, test_range, model, device, batch_size=4, epoches=40, name='model', drop_last=False):
    # During model training, we applied online data augmentation methods to input meibomian gland images
    Jitter = KAug.ColorJiggle(brightness=0.2, contrast=0.3, p=0.8, keepdim=True)
    Affine = KAug.RandomAffine(degrees=(-20, 20), translate=0.2, scale=(0.8, 1.3), padding_mode="border", p=1, keepdim=True)
    Elastic = KAug.RandomElasticTransform(sigma=(16, 16), alpha=(0.5, 0.5), p=0.5, keepdim=True)
    Vflip = KAug.RandomVerticalFlip(p=0.5, keepdim=True)
    Hflip = KAug.RandomHorizontalFlip(p=0.5, keepdim=True)
    train_data = ImageDataset("./datasets", train_range, preload=True, data_expension=True)
    if test_range is not None:
        val_data = ImageDataset("./datasets", test_range, preload=True, data_expension=False)
    else:
        val_data = None
    train_loader = torch.utils.data.DataLoader(train_data,shuffle=True, batch_size=batch_size, drop_last=drop_last)
    if val_data is not None:
        val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=1, drop_last=drop_last)
    CrossEntropy = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    # To prevent overfitting, a learning rate adjustment strategy was adopted, reducing the learning rate to half every 50 epochs.
    scheduler = StepLR(optimizer, step_size=train_config['decay_interval'], gamma=0.5)
    min_val_loss = float('inf')
    start_epoch = 0
    last_save_epoch = 0
    if os.path.exists(name + ".log"):
        with open(name + ".log", 'r') as f:
            lines = f.readlines()
            last_record = lines[-1]
            if "stop" in last_record:
                print("Early stopping detected... Stop training.")
                return
            start_epoch = int(last_record.split(',')[0])
            min_val_loss = getMinLoss(lines)
            last_save_epoch = start_epoch
    if start_epoch != 0 and os.path.exists(name + ".pkl"):
        model.load_state_dict(torch.load(name + ".pkl"))
        print("load model from epoch {}".format(start_epoch), "Min val loss:", min_val_loss)
    for epoch in range(start_epoch, epoches):
        stepCount = 0
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels, img_name = data
            optimizer.zero_grad()
            labels = labels.to(device)
            inputs = inputs.to(device)
            inputs = Jitter(inputs)
            inputs = Vflip(inputs)
            labels = Vflip(labels, params=Vflip._params)
            inputs = Hflip(inputs)
            labels = Hflip(labels, params=Hflip._params)
            inputs = Elastic(inputs)
            labels = Elastic(labels, params=Elastic._params)
            inputs = Affine(inputs)
            labels = Affine(labels, params=Affine._params)
            inputs = inputs[:, :1, :, :]
            labels = labels[:, 0, :, :].to(torch.long)
            outputs = model(inputs)
            loss = CrossEntropy(outputs, labels)
            loss.backward()
            optimizer.step()
            stepCount += 1
            running_loss += loss.item()
            if i % 10 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i, running_loss / stepCount))
                running_loss = 0.0
                stepCount = 0
            # if i % 100 == 99 or i == len(train_loader) - 1:
        print("Epoch", epoch + 1, "Time:", time.time() - start_time, "s")
        print("Start validating, image count:", train_data.__len__())
        scheduler.step()
        val_step_count = 0
        totalValLoss = 0
        for testitem in iter(val_loader):
            inputs, labels, img_name = testitem
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs[:, :1, :, :]
            labels = labels[:, 0, :, :].to(torch.long)
            result = model(inputs)
            loss = CrossEntropy(result, labels)
            totalValLoss += loss.item()
            val_step_count += 1
        val_loss = totalValLoss / val_step_count
        print("Val Loss:", val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            last_save_epoch = epoch
            torch.save(model.state_dict(), name + ".pkl")
            print("Save model to {}".format(name + ".pkl"))
        with open(name + ".log", 'a') as f:
            f.write(str(epoch) + "," + str(val_loss) + "\n")
        if epoch - last_save_epoch > 50:
            print("Early stoping...")
            with open(name + ".log", 'a') as f:
                f.write("stop\n")
            break


def cross_validation():
    # five-fold cross-validation
    device = torch.device('cuda:0')
    dataset_length = train_config['datasets_length']
    all_datasets = list(range(1, dataset_length))
    # specified validation set indices
    test_datasets = [
        list(range(1, 20)),
        # list(range(42, 83)),
        # list(range(83, 124)),
        # list(range(124, 165)),
        # list(range(165, 205))
    ]

    train_datasets = [
        [i for i in all_datasets if i not in val] for val in test_datasets
    ]

    if not os.path.exists("./weights"):
        os.mkdir("./weights")

    #==================demo U-Net=================================================================
    net = UNet(train_config['in_channels'], train_config['out_channels'])
    net = net.to(device)
    model_name = train_config['model_name']
    # model_name = './weights/model_unet_' + str(8)
    train(list(range(1, dataset_length)), None, net, device, epoches=train_config['num_epochs'], batch_size=train_config['batch_size'], name=model_name)
    save_result(net, model_name, list(range(1, dataset_length)), train_config['output_dir'])
    evaluate(net, model_name, list(range(1, dataset_length)))


if __name__ == "__main__":
    # load config file
    with open('train_config.yaml', 'r') as file:
        train_config = yaml.safe_load(file)
    cross_validation()
