# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Human-Pose-Estimation-Pipeline
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2020-10-09 10:52:59
# --------------------------------------------------------
"""
from torchvision import transforms
from pose.augment import augment_image


def train_transforms_v1():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomChoice([
        #     transforms.ColorJitter(brightness=0.5),
        #     transforms.ColorJitter(contrast=0.5),
        #     transforms.ColorJitter(saturation=0.5),
        #     transforms.ColorJitter(hue=0.5)]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
    ])
    return transform


def val_transforms_v1():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
    ])
    return transform


def train_transforms_v2():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        augment_image.RandomColorJitter(p=0.5,
                                        brightness=0.5,
                                        contrast=0.5,
                                        saturation=0.5,
                                        hue=0.1),
        transforms.RandomChoice([
            augment_image.RandomMotionBlur(degree=5, angle=360, p=0.5),
            augment_image.RandomGaussianBlur(ksize=(1, 1, 1, 3, 3, 5)),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform


def val_transforms_v2():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform


def train_transforms():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomChoice([
        #     transforms.ColorJitter(brightness=0.5),
        #     transforms.ColorJitter(contrast=0.5),
        #     transforms.ColorJitter(saturation=0.5),
        #     transforms.ColorJitter(hue=0.5)]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform


def val_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform
