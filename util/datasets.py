# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import pre_dataset as data2

def build_dataset2(is_train, args):


    if is_train:
        transform = transforms.Compose([
        #data2.RandomResizedCrop(224, scale=(0.99, 1), ratio=(0.99, 1.01), interpolation=3),
        data2.RandomResizedCrop(224, scale=(0.7, 1),ratio=(0.7, 1.2) ,interpolation=3),
        data2.RandomHorizontalFlip(),

        #data2.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.2, hue=0.05),
        #data2.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.2, hue=0.05),
        #transforms.GaussianBlur(kernel_size=(5, 11)),
        #transforms.GaussianBlur(kernel_size=(5, 11)),


        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.01, 0.1), ratio=(0.6,1.5), value=(0), inplace=False),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([
            transforms.Resize([224, 448]),  # 3 is bicubic
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)
    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    '''
    transform = transforms.Compose([
        transforms.Resize([224, 448]),  # 3 is bicubic
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    '''
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            hflip=0.5,
            vflip=0,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    #if args.input_size <= 224:
        #crop_pct = 224 / 256
    #else:
        #crop_pct = 1.0
    #size = int(args.input_size / crop_pct)

    t.append(
        transforms.Resize([224, 448], interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop((224,448)))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
