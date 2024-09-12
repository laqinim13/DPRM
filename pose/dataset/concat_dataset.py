# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-06-09 20:04:18
"""
import logging
import random
from .JointsDataset import JointsDataset
from torch.utils import data
import bisect
import math

logger = logging.getLogger(__name__)


class ConcatDataset(data.ConcatDataset):
    def __init__(self, datasets, ) -> None:
        super(ConcatDataset, self).__init__(datasets)
        self.flip_pairs = datasets[0].flip_pairs
        self.data_evaluate = datasets[0].evaluate
        
    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        return self.data_evaluate(cfg, preds, output_dir, all_boxes, img_path,
                                  *args, **kwargs)


class ConcatDatasetResample(data.ConcatDataset):
    def __init__(self, datasets, resample=False, balance=0.4, shuffle=False) -> None:
        super(ConcatDatasetResample, self).__init__(datasets)
        self.flip_pairs = datasets[0].flip_pairs
        self.data_evaluate = datasets[0].evaluate
        self.resample = resample
        if self.resample:
            self.balance = balance
            self.shuffle = shuffle
            self.person_pen_nums = [len(d) for d in datasets]
            self.nums = super().__len__()
            assert self.nums == sum(self.person_pen_nums)
            self.image_idx = self.resample_data(self.balance, self.shuffle)

    def resample_data(self, balance, shuffle=False):
        idx = list(range(self.nums))
        balance_nums = int(balance * self.nums)
        pen_idx = self.get_sampler(idx[:self.person_pen_nums[0]], balance_nums, shuffle)
        person_idx = self.get_sampler(idx[self.person_pen_nums[0]:self.nums], self.nums - balance_nums, shuffle)
        image_idx = pen_idx + person_idx
        if shuffle:
            random.shuffle(image_idx)
        logger.info("shuffle={},pen person:[{},{}]={}".format(shuffle, len(pen_idx), len(person_idx), len(image_idx)))
        return image_idx

    def __len__(self):
        if self.resample:
            self.image_idx = self.resample_data(self.balance, self.shuffle)
        return super().__len__()

    def get_sampler(self, item_list, nums, shuffle=True):
        """
        提取nums个数，不足nums个时，会进行填充
        :param item_list: 输入样本列表
        :param nums: 需要提取的样本数目
        :param shuffle: 是否随机提取样本
        :return:
        """
        item_nums = len(item_list)
        if nums > item_nums:
            item_list = item_list * math.ceil(nums / item_nums)
        if shuffle:
            random.shuffle(item_list)
        out_list = item_list[:nums]
        return out_list

    def __getitem__(self, idx):
        if self.resample:
            idx = self.image_idx[idx]
        return super().__getitem__(idx)

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        return self.data_evaluate(cfg, preds, output_dir, all_boxes, img_path,
                                  *args, **kwargs)


class CustomConcatDataset(JointsDataset):
    """ Concat Dataset """

    def __init__(self, datasets, shuffle=False):
        """
        import torch.utils.data as torch_utils
        voc1 = PolygonParser(filename1)
        voc2 = PolygonParser(filename2)
        voc=torch_utils.ConcatDataset([voc1, voc2])
        ====================================
        :param datasets:
        :param shuffle:
        """
        # super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'dataset should not be an empty iterable'
        # super(ConcatDataset, self).__init__()
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.image_id = []
        self.dataset = datasets
        self.shuffle = shuffle
        self.flip_pairs = []
        for dataset_id, dataset in enumerate(self.dataset):
            image_id = dataset.db
            image_id = self.add_dataset_id(image_id, dataset_id)
            self.image_id += image_id
            self.classes = dataset.classes
            self.flip_pairs = dataset.flip_pairs
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_id)

    def add_dataset_id(self, image_id, dataset_id):
        """
        :param image_id:
        :param dataset_id:
        :return:
        """
        out_image_id = []
        for id in image_id:
            out_image_id.append({"dataset_id": dataset_id, "image_id": id})
        return out_image_id

    def __getitem__(self, index):
        """
        :param index: int
        :return:
        """
        dataset_id = self.image_id[index]["dataset_id"]
        image_id = self.image_id[index]["image_id"]
        dataset = self.dataset[dataset_id]
        # print(dataset.data_root, image_id)
        data = dataset.__getitem__(image_id)
        return data

    def get_image_anno_file(self, index):
        dataset_id = index["dataset_id"]
        image_id = index["image_id"]
        return self.dataset[dataset_id].get_image_anno_file(image_id)

    def get_annotation(self, xml_file):
        return self.dataset[0].get_annotation(xml_file)

    def __len__(self):
        return len(self.image_id)
