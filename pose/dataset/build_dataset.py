# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-11-04 08:54:34
"""
import pose.dataset as dataset
from pose.dataset import data_transforms


def load_dataset(cfg, root, image_set, is_train, transform, shuffle=True):
    """
    加载数据
    :param cfg:
    :param root:
    :param image_set:
    :param is_train:
    :param transform:
    :param shuffle:
    :return:
    """
    if is_train:
        anns_file = cfg.DATASET.TRAIN_FILE
    else:
        anns_file = cfg.DATASET.TEST_FILE
        assert isinstance(anns_file, str), "测试数据只支持单个数据集"
    if cfg.DATASET.DATASET.lower() == "coco":
        data = dataset.coco(cfg, anns_file, image_set, is_train, transform)
        print("load dataset:{},have data:{}".format(anns_file, len(data)))
    elif cfg.DATASET.DATASET.lower() == "custom_coco":
        anns_file = [anns_file] if isinstance(anns_file, str) else anns_file
        data_list = []
        for ann_file in anns_file:
            data = dataset.custom_coco(cfg, ann_file, image_set, is_train, transform, shuffle=shuffle)
            print("load dataset:{},have data:{}".format(ann_file, len(data)))
            data_list.append(data)
        data = dataset.ConcatDataset(data_list)
    elif cfg.DATASET.DATASET.lower() == "mpii":
        data = dataset.mpii(cfg, anns_file, image_set, is_train, transform, shuffle=shuffle)
        print("load dataset:{},have data:{}".format(anns_file, len(data)))
    elif cfg.DATASET.DATASET.lower() == "custom_mpii":
        data = dataset.custom_mpii(cfg, anns_file, image_set, is_train, transform, shuffle=shuffle)
        print("load dataset:{},have data:{}".format(anns_file, len(data)))
    else:
        raise Exception("Error: no dataset:{}".format(cfg.DATASET.DATASET))
    return data
