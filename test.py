# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-11-08 15:02:19
"""
import argparse
import os
import torch.utils.data
from pose.core import function, function_offset
from pose.utils import setup_config, log
from pybaseutils import file_utils

import train

root = os.path.dirname(__file__)
print("torch:{}".format(torch.__version__))


class Validation(train.Trainer):
    def __init__(self, opt):
        self.cfg = self.parser_config(opt)
        self.is_main_process = True
        self.cfg.TEST.POST_PROCESS = True  # 设置True，可以提高一个点
        self.cfg.TEST.FLIP_TEST = True  # 设置True，可以提高一个点
        self.cfg.work_dir = os.path.join(os.path.dirname(self.cfg.model_file), "test")
        self.cfg.model_dir = os.path.join(self.cfg.work_dir, "model")
        self.cfg.log_dir = os.path.join(self.cfg.work_dir, "log")
        file_utils.create_dir(self.cfg.work_dir)
        file_utils.create_dir(self.cfg.model_dir)
        file_utils.create_dir(self.cfg.log_dir)
        self.logger = log.set_logger(level="debug",
                                     logfile=os.path.join(self.cfg.log_dir, "train.log"),
                                     is_main_process=self.is_main_process)
        self.cfg.finetune = self.cfg.model_file
        self.writer_dict = {}
        self.gpu_id = [int(i) for i in self.cfg.gpu_id.split(',')]
        self.device = "cuda:{}".format(self.gpu_id[0])
        self.out_type = self.cfg.MODEL.OUT_TYPE
        self.test_loader = self.build_test_loader(self.cfg)
        self.build(self.cfg)

    def build(self, cfg):
        self.model = self.build_model(cfg)
        self.criterion = self.build_criterion(cfg)

    def test(self, ):
        self.max_ap = 0.0
        if self.out_type == 'offset':
            self.train = function_offset.train
            self.validate = function_offset.validate
        else:
            self.train = function.train
            self.validate = function.validate
        self.model.eval()
        ap = self.run_test_epoch(0)
        self.logger.info('AP: {}'.format(ap))
        self.logger.info('model file: {}'.format(self.cfg.model_file))


def get_parser(opt=None):
    model_file = "work_space/hand/hrnet_w32_21_192_192_custom_coco_20231007_083128_2043/model/best_model_189_0.8570.pth"
    config_file = "work_space/hand/hrnet_w32_21_192_192_custom_coco_20231007_083128_2043/w32_adam_hand_192_192.yaml"
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument("-c", "--config_file", help="configs file", default=config_file, type=str)
    parser.add_argument('--batch_size', help='batch_size', default=32, type=int)
    parser.add_argument('--workers', help='workers', default=8, type=int)
    parser.add_argument('--model_file', help='model_file', type=str, default=model_file)
    parser.add_argument('--finetune', help='finetune', type=str, default="")
    parser.add_argument('--work_dir', help='work_dir', type=str, default="work_space/Test")
    parser.add_argument('--gpu_id', default="0", type=str, help='GPU ID')
    opt = setup_config.parser_config(parser.parse_args(), cfg_updata=True)
    return opt


if __name__ == '__main__':
    opt = get_parser()
    t = Validation(opt)
    t.test()
