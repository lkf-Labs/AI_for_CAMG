import os
import time

import kornia
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import random
from sklearn.metrics import jaccard_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from SegDataset import ImageDataset



def showImg(img):
    img = img.squeeze()
    if img.shape[0] < 5:
        img = img.transpose(0, 1)
        img = img.transpose(1, 2)
    plt.imshow(img, cmap="gray")
    plt.show()


def plot(figure, data, net, idxFrom, label):
    width, height = 4, 4
    randIdx = random.randint(0, data.__len__()-1)
    img, seg, name = data.__getitem__(randIdx)
    img = img.unsqueeze(0)
    print("input shape", img.shape)
    net.eval()
    result, _ = net(img)
    img_show = img.squeeze()
    img_show = img_show.transpose(0, 1)
    img_show = img_show.transpose(1, 2)
    print("result shape", result.shape)

    figure.add_subplot(width, height, 1+idxFrom*4)
    plt.title(label + " Image " + data.ImgNames[randIdx])
    plt.axis("off")
    plt.imshow(img_show)
    figure.add_subplot(width, height, 2+idxFrom*4)
    plt.title("Ground Truth")
    plt.axis("off")
    plt.imshow(seg.squeeze(), cmap="gray")
    figure.add_subplot(width, height, 3+idxFrom*4)
    plt.title("Predict")
    plt.axis("off")
    plt.imshow(result.detach().squeeze(), cmap="gray")
    figure.add_subplot(width, height, 4+idxFrom*4)
    plt.title("Predict Binarized")
    plt.axis("off")
    result = nn.Sigmoid()(result)
    plt.imshow(result.detach().squeeze() > 0.5, cmap="gray")


def display(net, model_path, data_range):
    UseGPU = True
    device = torch.device('cuda:0')
    if UseGPU:
        net = net.to(device)
    train_data = ImageDataset("./datasets/train", data_range)
    val_data = ImageDataset("./datasets/val", data_range)

    BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    while True:
        if os.path.exists(model_path):
            print(model_path)
            net.load_state_dict(torch.load(model_path))
        figure = plt.figure(figsize=(12, 8))
        plot(figure, train_data, net, 0, "Train")
        plot(figure, train_data, net, 1, "Train")
        plot(figure, val_data, net, 2, "Val")
        plot(figure, val_data, net, 3, "Val")
        plt.show()
        input()


def save_result(net, model_path, data_range, save_path="./result"):
    print("Evaluating", model_path)
    device = torch.device("cuda:0")
    UseGPU = True
    if UseGPU:
        net = net.to(device)
    net.load_state_dict(torch.load(model_path + ".pkl"))
    net.eval()
    val_data = ImageDataset("./datasets", data_range)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1)
    seg_list = []
    pred_list = []
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for idx, item in enumerate(val_loader):
        img, seg, name = item
        if UseGPU:
            img = img.to(device)
        img = img[:, :1, :, :]
        st = time.time()
        seg_out = net(img)
        print("time:", time.time() - st)
        seg_out = seg_out.detach().cpu()
        seg_out = torch.argmax(seg_out, dim=1)
        seg_out = seg_out.squeeze().to(torch.float)
        save_image(seg_out, save_path + "/" + name[0] + ".png")


def evaluate(net, model_path, data_range):
    print("Evaluating", model_path)
    device = torch.device("cuda:0")
    UseGPU = True
    if UseGPU:
        net = net.to(device)
    print(model_path)
    net.load_state_dict(torch.load(model_path + ".pkl"))
    net.eval()
    val_data = ImageDataset("./datasets", data_range)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1)
    seg_list = []
    pred_list = []
    for idx, item in enumerate(val_loader):
        img, seg, name = item
        if UseGPU:
            img = img.to(device)
        img = img[:, :1, :, :]
        seg = seg[:, 0, :, :].to(torch.long)
        seg_out = net(img)
        seg_out = seg_out.detach().cpu()
        seg_out = torch.argmax(seg_out, dim=1)
        seg = seg.squeeze().to(torch.float)
        seg_list.append(seg)
        pred_list.append(seg_out.squeeze())
        print(idx, end='')
    for i in range(len(seg_list)):
        if seg_list[i].dim() == 2:
            seg_list[i] = seg_list[i].unsqueeze(0)
            pred_list[i] = pred_list[i].unsqueeze(0)
    segs = torch.cat(seg_list).numpy().flatten()
    preds = torch.cat(pred_list).numpy().flatten()
    print("Cat Done")
    print(segs, preds)
    segs = segs.astype(np.uint8)
    IOU = jaccard_score(segs, preds, average=None)
    print("IOU Done")
    acc = accuracy_score(segs, preds)
    print("Acc Done")
    precision = precision_score(segs, preds)
    print("Precision Done")
    recall = recall_score(segs, preds)
    print("Recall Done")
    f1 = f1_score(segs, preds)
    print("F1 Done")
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("IOU: ", IOU)
    # write the data into file
    with open("result_seg.txt", "a") as f:
        f.write("Model: " + model_path + "\n")
        f.write("Accuracy: " + str(acc) + "\n")
        f.write("Precision: " + str(precision) + "\n")
        f.write("Recall: " + str(recall) + "\n")
        f.write("F1: " + str(f1) + "\n")
        f.write("IOU: " + str(IOU) + "\n")
    return acc, precision, recall, f1, IOU



if __name__ == "__main__":
    net = NestedUNet(None, 1, 2)
    save_result(net, "./weights/model_unetpp_0", list(range(1, 40)))