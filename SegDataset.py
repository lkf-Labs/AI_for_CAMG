import os
import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
import kornia as K
import kornia.augmentation as KAug
import random


class ImageDatasetWithRandomExpand(Dataset):
    def __init__(self, path, data_range, channels=1, width=736, height=352, preload=False, data_expension=False):
        self.ImgNames = []
        self.Imgs = []
        self.Segs = []
        self.Path = path
        self.Preload = preload
        self.Channels = channels
        self.Width = width
        self.Height = height
        self.DataExpension = data_expension
        self.Padding = T.Pad(1, padding_mode="edge")
        self.Jitter = KAug.ColorJiggle(brightness=0.2, contrast=0.3, p=0.8, keepdim=True)
        self.Affine = KAug.RandomAffine(degrees=(-20, 20), translate=0.2, scale=(0.8, 1.3), padding_mode="border", p=1, keepdim=True)
        self.Elastic = KAug.RandomElasticTransform(sigma=(16, 16), alpha=(0.5, 0.5), p=0.5, keepdim=True)
        self.Vflip = KAug.RandomVerticalFlip(p=0.5, keepdim=True)
        self.Hflip = KAug.RandomHorizontalFlip(p=0.5, keepdim=True)
        fullPath = os.path.join(path, "img")
        count = 0
        for file in os.listdir(fullPath):
            if "_" in file:
                args = file.split("_")
                idx = int(args[0])
                expend_idx = int(args[1])
                if expend_idx > 5:
                    continue
            else:
                idx = int(file.split(".")[0])
            if idx in data_range:
                file_name = file.split(".")[0]
                self.ImgNames.append(file_name)
                count += 1
                if count % 100 == 0:
                    print("{} images loaded".format(count))
        print("Loaded {} images list".format(count))
        print("Preloading images...")
        if self.Preload:
            for i in range(self.__len__()):
                img, seg = self.preload_image(i)
                self.Imgs.append(img)
                self.Segs.append(seg)
                if i % 100 == 0 or i == self.__len__() - 1:
                    print("{} images preloaded".format(i))

    def __len__(self):
        return len(self.ImgNames)

    def __getitem__(self, index):
        if self.Preload:
            img, seg = self.Imgs[index], self.Segs[index]
        else:
            img, seg = self.preload_image(index)

        return img, seg, self.ImgNames[index]

    def preload_image(self, index):
        imgPath = os.path.join(self.Path, "img", self.ImgNames[index] + ".png")
        segPath = os.path.join(self.Path, "seg", self.ImgNames[index] + ".png")
        img = read_image(imgPath, ImageReadMode.RGB)
        if os.path.exists(segPath):
            seg = read_image(segPath, ImageReadMode.RGB)
        else:
            seg = read_image(segPath.replace(".png", ".gif"), ImageReadMode.RGB)
        img = img / 255.0
        seg = seg / 255.0
        img = self.Padding(img)
        seg = self.Padding(seg)
        img = T.functional.crop(img, 0, 0, self.Height, self.Width)
        seg = T.functional.crop(seg, 0, 0, self.Height, self.Width)
        return img, seg


class ImageDatasetLagecy(Dataset):
    def __init__(self, path, data_range):
        self.ImgNames = []
        self.Imgs = []
        self.Segs = []
        self.Path = path
        self.Preload = False
        self.resize = T.Resize((352, 736))
        fullPath = os.path.join(path, "img")
        count = 0
        for file in os.listdir(fullPath):
            if "_" in file:
                args = file.split("_")
                idx = int(args[0])
                expend_idx = int(args[1])
                if expend_idx > 5:
                    continue
            else:
                idx = int(file.split(".")[0])
            if idx in data_range:
                file_name = file.split(".")[0]
                if self.Preload:
                    imgPath = os.path.join(self.Path, "img", file_name + ".png")
                    segPath = os.path.join(self.Path, "seg", file_name + ".png")
                    if os.path.exists(imgPath):
                        self.Imgs.append(read_image(imgPath))
                    if os.path.exists(segPath):
                        self.Segs.append(read_image(segPath))
                self.ImgNames.append(file_name)
                count += 1
                if count % 100 == 0:
                    print("{} images loaded".format(count))
        print("Loaded {} images".format(count))

    def __len__(self):
        return len(self.ImgNames)

    def __getitem__(self, index):
        if self.Preload:
            img = self.Imgs[index]
            seg = self.Segs[index]
        else:
            imgPath = os.path.join(self.Path, "img", self.ImgNames[index] + ".png")
            segPath = os.path.join(self.Path, "seg", self.ImgNames[index] + ".png")
            img = read_image(imgPath)
            if os.path.exists(segPath):
                seg = read_image(segPath)
            else:
                seg = read_image(segPath.replace(".png", ".gif"))
        img = self.resize(img)
        seg = self.resize(seg)
        img = T.ColorJitter(contrast=(2,2))(img[:-1,:,:]/255)[:1,:,:]
        seg = torch.round(seg[0, :, :] / 255)
        seg = seg.to(torch.long)
        return img, seg, self.ImgNames[index]


class ImageDatasetWithROI(Dataset):
    def __init__(self, path, data_range, channels=1, width=736, height=352, preload=False, data_expension=False):
        self.ImgNames = []
        self.Imgs = []
        self.Segs = []
        self.ROIs = []
        self.Path = path
        self.Preload = preload
        self.Channels = channels
        self.Width = width
        self.Height = height
        self.DataExpension = data_expension
        self.Padding = T.Pad(1, padding_mode="edge")
        fullPath = os.path.join(path, "image_crop")
        count = 0
        for file in os.listdir(fullPath):
            if "_" in file:
                args = file.split("_")
                idx = int(args[0])
                expend_idx = int(args[1])
                if expend_idx > 5:
                    continue
            else:
                idx = int(file.split(".")[0])
            if idx in data_range:
                file_name = file.split(".")[0]
                self.ImgNames.append(file_name)
                count += 1
                if count % 100 == 0:
                    print("{} images loaded".format(count))
        print("Loaded {} images list".format(count))
        print("Preloading images...")
        if self.Preload:
            for i in range(self.__len__()):
                img, seg = self.preload_image(i)
                self.Imgs.append(img)
                self.Segs.append(seg)
                if i % 100 == 0 or i == self.__len__() - 1:
                    print("{} images preloaded".format(i))

    def __len__(self):
        return len(self.ImgNames)

    def __getitem__(self, index):
        if self.Preload:
            img, seg, roi = self.Imgs[index], self.Segs[index]
        else:
            img, seg, roi = self.preload_image(index)
        # img = T.functional.adjust_contrast(img, 1.2)
        # if self.DataExpension:
        #     img = self.Vflip(img)
        #     seg = self.Vflip(seg, params=self.Vflip._params)
        #     img = self.Hflip(img)
        #     seg = self.Hflip(seg, params=self.Hflip._params)
        #     img = self.Elastic(img)
        #     seg = self.Elastic(seg, params=self.Elastic._params)
        #     img = self.Affine(img)
        #     seg = self.Affine(seg, params=self.Affine._params)
        # img = img[:1,:,:]
        # seg = torch.round(seg[0, :, :])
        # seg = seg.to(torch.long)
        return img, seg, roi, self.ImgNames[index]

    def preload_image(self, index):
        imgPath = os.path.join(self.Path, "image_crop", self.ImgNames[index] + ".png")
        segPath = os.path.join(self.Path, "label_crop", self.ImgNames[index] + ".png")
        roiPath = os.path.join(self.Path, "roi_crop", self.ImgNames[index] + ".png")
        img = read_image(imgPath, ImageReadMode.RGB)
        roi = read_image(roiPath, ImageReadMode.RGB)
        if os.path.exists(segPath):
            seg = read_image(segPath, ImageReadMode.RGB)
        else:
            seg = read_image(segPath.replace(".png", ".gif"), ImageReadMode.RGB)
        img = img / 255.0
        seg = seg / 255.0
        roi = roi / 255.0
        img = self.Padding(img)
        seg = self.Padding(seg)
        roi = self.Padding(roi)
        img = T.functional.crop(img, 0, 0, self.Height, self.Width)
        seg = T.functional.crop(seg, 0, 0, self.Height, self.Width)
        roi = T.functional.crop(roi, 0, 0, self.Height, self.Width)
        return img, seg, roi


ImageDataset = ImageDatasetWithRandomExpand


# class ImageDataset(Dataset):
#     def __init__(self, path, data_range, expend_range=None, channels=3, width=736, height=352):
#         self.ImgNames = []
#         self.Imgs = []
#         self.Segs = []
#         self.Path = path
#         self.Preload = False
#         self.Channels = channels
#         self.Width = width
#         self.Height = height
#         fullPath = os.path.join(path, "img")
#         count = 0
#         for file in os.listdir(fullPath):
#             if "_" in file:
#                 args = file.split("_")
#                 idx = int(args[0])
#                 expend_idx = int(args[1])
#                 if expend_range is not None and expend_idx not in expend_range:
#                     continue
#             else:
#                 idx = int(file.split(".")[0])
#             if idx in data_range:
#                 file_name = file.split(".")[0]
#                 self.ImgNames.append(file_name)
#                 count += 1
#                 if count % 100 == 0:
#                     print("{} images loaded".format(count))
#         print("Loaded {} images".format(count))
#
#     def __len__(self):
#         return len(self.ImgNames)
#
#     def __getitem__(self, index):
#         if self.Preload:
#             img = self.Imgs[index]
#             seg = self.Segs[index]
#         else:
#             imgPath = os.path.join(self.Path, "img", self.ImgNames[index] + ".png")
#             segPath = os.path.join(self.Path, "seg", self.ImgNames[index] + ".png")
#             img = read_image(imgPath)
#             if os.path.exists(segPath):
#                 seg = read_image(segPath)
#             else:
#                 seg = read_image(segPath.replace(".png", ".gif"))
#         # print(img.shape)
#         img = transforms.Resize((self.Height, self.Width))(img)
#         seg = transforms.Resize((self.Height, self.Width))(seg)
#         if self.Channels == 1:
#             img = img[:1,:,:] / 255
#         else:
#             img = img[:3,:,:] / 255
#         seg = seg[0,:,:] / 255
#         seg = torch.unsqueeze(seg, 0)
#         # img = transforms.ColorJitter(contrast=(2,2))(img[:-1,:,:]/255)[0,:,:]
#         # seg = seg[0,:,:]/255
#         # img = torch.unsqueeze(img, 0)
#         # seg = torch.unsqueeze(seg, 0)
#         return img, seg, self.ImgNames[index]
