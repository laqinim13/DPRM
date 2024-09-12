# -*-coding: utf-8 -*-
"""
    @Project: torch-Human-Pose-Estimation-Pipeline
    @File   : train.py
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-11-08 15:02:19
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from pose.dataset import data_transforms
import pose.dataset as dataset
from tensorboardX import SummaryWriter
from pose.config.config import config
from pose.nets.tf_build_nets import build_nets
from pose.config.config import update_config
from pose.tf_core import tf_loss
from pose.tf_core.tf_function import train, validate
from pose.tf_core.tf_multistep_lr import MultiStepLR
from pose.utils.utils import create_work_space
import tensorflow as tf
from utils import file_utils
from utils import tf_tools
from pose.utils import torch_tools

print("TF:{}".format(tf.__version__))


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # default="./configs/coco/resnet50/train_res18.yaml",
                        # default="./configs/coco/resnet50/train_res18_body.yaml",
                        # default="./configs/coco/resnet50/train_res18_finger.yaml",
                        # default="./configs/coco/resnet50/train_res18_finger_tf.yaml",
                        # default="./configs/coco/resnet50/train_res18_Bottleneck.yaml",
                        default="./configs/coco/mobilenet/local_model_mbv2_finger_tf.yaml",
                        # default="./configs/coco/resnet50/train_res50_finger4_tf.yaml",
                        # default="./configs/mpii/resnet50/train.yaml",
                        # default="./configs/mpii/resnet50/train_student.yaml",
                        # default="./configs/mpii/mobilenet/train_student_mbv2.yaml",
                        # default="./configs/mpii/mobilenet/train_mbv2.yaml",
                        # default="./configs/mpii/mobilenet/train_ir_mbv2.yaml",
                        # default="./configs/mpii/mobilenet/train_model_mbv2.yaml",
                        # default="./configs/mpii/mobilenet/train_body_mbv2.yaml",
                        # default="./configs/mpii/mobilenet/train_xionghao_mbv2.yaml",
                        # default="./configs/mpii/mobilenet/train_model_ir_mbv2.yaml",
                        type=str)
    args, rest = parser.parse_known_args()
    # update config#
    update_config(args.cfg)
    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.PRINT_FREQ, type=int)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--flag', help='flag', type=str, default="")
    parser.add_argument('--work_dir', help='work_dir', type=str, default="work_dir/Test")
    parser.add_argument("--tf_log_level", help="set TF_CPP_MIN_LOG_LEVEL", default="1", type=str)
    parser.add_argument('--gpu_id', default="0", type=str, help='path to dataset')
    parser.add_argument('--polyaxon', action='store_true', help='polyaxon', default=False)
    args = parser.parse_args()
    if args.polyaxon:
        from pose.utils import rsync_data
        print("use polyaxon")
        config.DATASET.ROOT = "dataset/dataset/finger_keypoint/finger"
        config.DATASET.ROOT = rsync_data.get_polyaxon_dataroot(root="ceph", dir=config.DATASET.ROOT)
        args.work_dir = os.path.join(rsync_data.get_polyaxon_output(), args.work_dir)
    return args


class Trainer():
    def __init__(self, args):
        self.args = args
        torch_tools.set_env_random_seed()
        tf_tools.set_env_random_seed()
        tf_tools.set_device_memory()

        config.OUTPUT_DIR = args.work_dir
        self.logger, self.work_dir, self.tb_log_dir = create_work_space(config, args.cfg, args.flag, phase='train')
        self.model_dir = os.path.join(self.work_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.logger.info(pprint.pformat(args))
        self.logger.info(pprint.pformat(config))

        self.writer_dict = {
            'writer': SummaryWriter(logdir=self.tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
        self.OUT_TYPE = config.MODEL.EXTRA.OUT_TYPE
        self.load_train_val_data()
        self.build()

    def build(self):
        self.model = build_nets(net_type=config.MODEL.NAME, config=config, is_train=True)
        # define loss function (criterion) and optimizer
        self.lr_scheduler = MultiStepLR(lr_stages=config.TRAIN.LR_STEP,
                                        init_lr=config.TRAIN.LR,
                                        steps_per_epoch=0,
                                        warmup_epoch=0,
                                        multi_gpu=False)
        if self.OUT_TYPE == 'gaussian':
            # self.criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT)
            # self.criterion = tf_loss.JointsMSE_Loss1
            self.criterion = tf_loss.JointsMSE_Loss2
        elif self.OUT_TYPE == 'offset':
            self.criterion = None
        else:
            raise Exception("Error:{}".format(self.OUT_TYPE))
        self.optimizer = tf.optimizers.Adam(learning_rate=config.TRAIN.LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        # self.optimizer = tf.optimizers.Adam(learning_rate=config.TRAIN.LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    def load_train_val_data(self, ):
        # train_transforms = data_transforms.train_transforms()
        # val_transforms = data_transforms.val_transforms()
        train_transforms = data_transforms.train_transforms_v2()
        val_transforms = data_transforms.val_transforms_v2()

        if config.DATASET.DATASET == "coco":
            self.train_dataset = dataset.coco(config, config.DATASET.ROOT, config.DATASET.TRAIN_SET, True,
                                              train_transforms)
            self.valid_dataset = dataset.coco(config, config.DATASET.ROOT, config.DATASET.TEST_SET, False,
                                              val_transforms)
        elif config.DATASET.DATASET == "custom_coco":
            self.train_dataset = dataset.custom_coco(config, config.DATASET.ROOT, config.DATASET.TRAIN_SET, True,
                                                     train_transforms)
            self.valid_dataset = dataset.custom_coco(config, config.DATASET.ROOT, config.DATASET.TEST_SET, False,
                                                     val_transforms)

        elif config.DATASET.DATASET == "mpii":
            self.train_dataset = dataset.mpii(config, config.DATASET.ROOT, config.DATASET.TRAIN_SET, True,
                                              train_transforms)
            self.valid_dataset = dataset.mpii(config, config.DATASET.ROOT, config.DATASET.TEST_SET, False,
                                              val_transforms)
        elif config.DATASET.DATASET == "custom_mpii":
            self.train_dataset = dataset.custom_mpii(config, config.DATASET.ROOT, config.DATASET.TRAIN_SET, True,
                                                     train_transforms)
            self.valid_dataset = dataset.custom_mpii(config, config.DATASET.ROOT, config.DATASET.TEST_SET, False,
                                                     val_transforms)
        elif config.DATASET.DATASET == "student_mpii":
            self.train_dataset = dataset.student_mpii(config, config.DATASET.ROOT, config.DATASET.TRAIN_SET, True,
                                                      train_transforms)
            self.valid_dataset = dataset.student_mpii(config, config.DATASET.ROOT, config.DATASET.TEST_SET, False,
                                                      val_transforms)
        else:
            raise Exception("Error: no dataset:{}".format(config.DATASET.DATASET))

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=False
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=False
        )

    def start_train(self, ):
        best_perf = 0.0
        best_model = False
        self.max_acc = 0.0
        self.lr_scheduler.set_model(self.model)
        self.model.optimizer = self.optimizer  # BUG: Model' object has no attribute 'loss'
        self.lr_scheduler.on_train_begin()
        for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
            self.logger.info('work space: {}'.format(self.work_dir))
            self.lr_scheduler.on_epoch_begin(epoch)
            # train for one epoch : image is BGR image
            train(config, self.train_loader, self.model, self.criterion, self.optimizer, epoch,
                  self.work_dir, self.tb_log_dir, self.writer_dict)
            # evaluate on validation set
            ap_value = validate(config, self.valid_loader, self.valid_dataset, self.model,
                                self.criterion, self.work_dir, self.tb_log_dir,
                                self.writer_dict, epoch)
            writer = self.writer_dict['writer']
            lr = tf.keras.backend.get_value(self.optimizer.lr)
            writer.add_scalar('lr_epoch', lr, epoch)

            if ap_value > best_perf:
                best_perf = ap_value
                best_model = True
            else:
                best_model = False

            self.logger.info('=> saving checkpoint to {}'.format(self.model_dir))
            self.logger.info('AP: {}'.format(ap_value))
            self.save_model(self.model_dir, config.MODEL.NAME, ap_value, epoch)
            # self.model.save_weights(filepath=os.path.join(self.model_dir, "model"), save_format="tf")
        # final_model_state_file = os.path.join(self.model_dir, 'final_state.pth.tar')
        # self.logger.info('saving final model state to {}'.format(final_model_state_file))
        self.writer_dict['writer'].close()

    def save_model(self, model_root, net_type, ap_value, epoch, start_save=0):
        """
        :param model_root:
        :param net_type:
        :param ap_value:
        :param epoch:
        :param start_save:
        :return:
        """
        # model = self.model
        # optimizer = self.optimizer
        start_save = start_save if start_save else config.TRAIN.END_EPOCH - 10
        if epoch >= start_save:
            model_file = os.path.join(model_root, "model_{}_{:0=3d}_{:.4f}".format(net_type, epoch, ap_value))
            # self.model.save_weights(filepath=model_file, save_format="tf")
            self.model.save(filepath=model_file, include_optimizer=False
                            )
            # self.model.load_weights()
            self.logger.info("save model in:{}".format(model_file))

        if self.max_acc <= ap_value:
            self.max_acc = ap_value
            best_model_file = os.path.join(model_root, "best_model_{:0=3d}_{:.4f}".format(epoch, ap_value))
            file_utils.remove_prefix_files(model_root, "best_model_*")
            # self.model.save_weights(filepath=best_model_file, save_format="tf")
            self.model.save(filepath=best_model_file, include_optimizer=False)
            self.logger.info("save best_model_path in:{}".format(best_model_file))


if __name__ == '__main__':
    # cudnn related setting
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = args.tf_log_level
    t = Trainer(args)
    t.start_train()
