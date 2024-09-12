# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-11-08 15:02:19
"""
import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import traceback
from tensorboardX import SummaryWriter
from pose.dataset import data_transforms, build_dataset
from pose.nets.build_nets import build_nets
from pose.core.loss import JointsMSELoss, JointsMSELoss_offset
from pose.core import function, function_offset
from pose.utils.utils import create_work_space
from pose.utils import setup_config, log
from basetrainer.utils import torch_tools
from pybaseutils import file_utils

root = os.path.dirname(__file__)
print("torch:{}".format(torch.__version__))


class Trainer(object):
    def __init__(self, opt):
        torch_tools.set_env_random_seed()
        self.root = root
        self.is_main_process = True
        self.cfg = self.parser_config(opt)
        self.cfg.work_dir = create_work_space(self.cfg, full_name=False)
        self.cfg.model_dir = os.path.join(self.cfg.work_dir, "model")
        self.cfg.log_dir = os.path.join(self.cfg.work_dir, "log")
        if self.is_main_process:
            file_utils.create_dir(self.cfg.work_dir)
            file_utils.create_dir(self.cfg.model_dir)
            file_utils.create_dir(self.cfg.log_dir)
            file_utils.copy_file_to_dir(self.cfg.config_file, self.cfg.work_dir)
            setup_config.save_config(self.cfg, os.path.join(self.cfg.work_dir, "setup_config.yaml"))
        self.logger = log.set_logger(level="debug",
                                     logfile=os.path.join(self.cfg.log_dir, "train.log"),
                                     is_main_process=self.is_main_process)
        self.gpu_id = [int(i) for i in self.cfg.gpu_id.split(',')]
        self.device = "cuda:{}".format(self.gpu_id[0])
        self.writer = SummaryWriter(log_dir=self.cfg.log_dir)
        self.writer_dict = {'writer': self.writer, 'train_global_steps': 0, 'valid_global_steps': 0}
        self.out_type = self.cfg.MODEL.OUT_TYPE
        self.train_loader = self.build_train_loader(self.cfg)
        self.test_loader = self.build_test_loader(self.cfg)
        self.build(self.cfg)
        self.logger.info("=" * 60)
        self.logger.info("work_dir          :{}".format(self.cfg.work_dir))
        self.logger.info("config_file       :{}".format(self.cfg.config_file))
        self.logger.info("image size(W,H)   :{}".format(self.cfg.MODEL.IMAGE_SIZE))
        self.logger.info("num joints        :{}".format(self.cfg.MODEL.NUM_JOINTS))
        self.logger.info("batch size        :{}".format(self.cfg.TRAIN.BATCH_SIZE))
        self.logger.info("model name        :{}".format(self.cfg.MODEL.NAME))
        self.logger.info("out_type          :{}".format(self.out_type))
        self.logger.info("gpu_id            :{}".format(self.gpu_id))
        self.logger.info("main device       :{}".format(self.device))
        self.logger.info("num_samples(train):{}".format(self.num_samples))
        self.logger.info("image size        :{}".format(self.cfg.MODEL.IMAGE_SIZE))
        self.logger.info("=" * 60)

    def parser_config(self, opt):
        cfg = setup_config.parser_config_file(opt, os.path.join(root, "pose/config/default.yaml"), True)
        cfg.WORKERS = opt.workers
        cfg.TRAIN.BATCH_SIZE = opt.batch_size
        cfg.TEST.BATCH_SIZE = opt.batch_size
        return cfg

    def build(self, cfg):
        self.model = self.build_model(cfg)
        self.criterion = self.build_criterion(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                 cfg.TRAIN.LR_STEP,
                                                                 cfg.TRAIN.LR_FACTOR)

    def build_criterion(self, cfg):
        """define loss function (criterion)"""
        self.logger.info("build_criterion")
        if self.out_type == 'gaussian':
            criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)
        elif self.out_type == 'offset':
            criterion = JointsMSELoss_offset(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)
        else:
            raise Exception("Error:{}".format(self.out_type))
        criterion = criterion.to(self.device)
        return criterion

    def build_model(self, cfg):
        """define model"""
        self.logger.info("build_model,net_type:{}".format(cfg.MODEL.NAME))
        model = build_nets(net_type=cfg.MODEL.NAME, config=cfg, is_train=True)
        if cfg.finetune:
            self.logger.info("finetune model:{}".format(cfg.finetune))
            model = torch_tools.load_pretrained_model(model, cfg.finetune)
        model = torch.nn.DataParallel(model, device_ids=self.gpu_id)
        model = model.to(self.device)
        return model

    def build_optimizer(self, cfg, model):
        """ define train optimizer"""
        self.logger.info("build_optimizer")
        self.logger.info("optim_type:{},init_lr:{},weight_decay:{}".format(cfg.TRAIN.OPTIMIZER,
                                                                           cfg.TRAIN.LR,
                                                                           cfg.TRAIN.WD))
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=cfg.TRAIN.LR,
                                        momentum=cfg.TRAIN.MOMENTUM,
                                        weight_decay=cfg.TRAIN.WD,
                                        nesterov=cfg.TRAIN.NESTEROV
                                        )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
            # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)
        elif cfg.TRAIN.OPTIMIZER == 'adamw':
            # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)
        else:
            raise Exception("Error:{}".format(cfg.TRAIN.OPTIMIZER))
        return optimizer

    def build_train_loader(self, cfg):
        """ define train dataset"""
        self.logger.info("build_train_loader")
        # transform = data_transforms.train_transforms()
        transform = data_transforms.train_transforms_v2()
        dataset = build_dataset.load_dataset(cfg,
                                             cfg.DATASET.ROOT,
                                             cfg.DATASET.TRAIN_SET,
                                             is_train=True,
                                             transform=transform,
                                             shuffle=True)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg.TRAIN.BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=cfg.WORKERS,
                                             pin_memory=False,
                                             drop_last=True
                                             )
        self.num_samples = len(dataset)
        self.logger.info("have train images {}".format(self.num_samples))
        return loader

    def build_test_loader(self, cfg):
        """ define test dataset"""
        self.logger.info("build_test_loader")
        # transform = data_transforms.val_transforms()
        transform = data_transforms.val_transforms_v2()
        dataset = build_dataset.load_dataset(cfg,
                                             cfg.DATASET.ROOT,
                                             cfg.DATASET.TEST_SET,
                                             is_train=False,
                                             transform=transform,
                                             shuffle=False)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg.TEST.BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=cfg.WORKERS,
                                             pin_memory=False
                                             )
        self.logger.info("have valid images {}".format(len(dataset)))
        return loader

    def run_train_epoch(self, epoch):
        # set to training mode
        self.model.train()
        self.train(self.cfg, self.train_loader, self.model, self.criterion, self.optimizer,
                   self.writer_dict, epoch, self.device)
        self.lr_scheduler.step()

    def run_test_epoch(self, epoch):
        # set to evaluates mode
        self.model.eval()
        ap = self.validate(self.cfg, self.test_loader, self.model, self.criterion,
                           self.writer_dict, epoch, self.device)
        return ap

    def run(self, ):
        self.max_ap = 0.0
        if self.out_type == 'offset':
            self.train = function_offset.train
            self.validate = function_offset.validate
        else:
            self.train = function.train
            self.validate = function.validate
        self.logger.info('target: {}'.format(self.out_type))
        for epoch in range(self.cfg.TRAIN.BEGIN_EPOCH, self.cfg.TRAIN.END_EPOCH):
            # train for one epoch : image is BGR image
            self.logger.info('epoch:{},work space: {}'.format(epoch, self.cfg.work_dir))
            self.run_train_epoch(epoch)
            ap = self.run_test_epoch(epoch)
            self.writer_dict['writer'].add_scalar('lr_epoch', self.optimizer.param_groups[0]['lr'], epoch)
            self.logger.info('=> saving checkpoint to {}'.format(self.cfg.model_dir))
            self.logger.info('AP: {}'.format(ap))
            self.save_model(self.cfg.model_dir, ap, epoch)

    def save_model(self, model_dir, ap, epoch, start_save=0):
        """
        :param model_dir:
        :param ap:
        :param epoch:
        :param start_save:
        :return:
        """
        model = self.model
        optimizer = self.optimizer
        model_file = os.path.join(model_dir, "model_{:0=3d}_{:.4f}.pth".format(epoch, ap))
        optimizer_pth = os.path.join(model_dir, "model_optimizer.pth")
        torch.save({"epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict()}, optimizer_pth)

        start_save = start_save if start_save else self.cfg.TRAIN.END_EPOCH - 10
        if epoch >= start_save:
            torch.save(model.module.state_dict(), model_file)
            self.logger.info("save model in:{}".format(model_file))

        if self.max_ap <= ap:
            self.max_ap = ap
            best_model_file = os.path.join(model_dir, "best_model_{:0=3d}_{:.4f}.pth".format(epoch, ap))
            file_utils.remove_prefix_files(model_dir, "best_model_*")
            torch.save(model.module.state_dict(), best_model_file)
            self.logger.info("save best_model_path in:{}".format(best_model_file))


def get_parser(opt=None):
    cfg = "configs/coco/hrnet/w32_adam_hand_192_192.yaml"
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument("-c", "--config_file", help="configs file", default=cfg, type=str)
    parser.add_argument('--batch_size', help='batch_size', default=16, type=int)
    parser.add_argument('--workers', help='workers', default=8, type=int)
    parser.add_argument('--flag', help='flag', type=str, default="")
    parser.add_argument('--finetune', help='finetune', type=str, default="")
    parser.add_argument('--work_dir', help='work_dir', type=str, default="work_space/Test")
    parser.add_argument('--gpu_id', default="0", type=str, help='GPU ID')
    parser.add_argument('--polyaxon', action='store_true', help='polyaxon', default=False)
    opt = setup_config.parser_config(parser.parse_args(), cfg_updata=True)
    if opt.polyaxon:
        from pose.utils import rsync_data
        print("use polyaxon")
        opt.DATASET.TRAIN_FILE = rsync_data.get_polyaxon_dataroot(opt.DATASET.TRAIN_FILE)
        opt.DATASET.TEST_FILE = rsync_data.get_polyaxon_dataroot(opt.DATASET.TEST_FILE)
        opt.work_dir = rsync_data.get_polyaxon_output(opt.work_dir)
    return opt


if __name__ == '__main__':
    opt = get_parser()
    t = Trainer(opt)
    t.run()
