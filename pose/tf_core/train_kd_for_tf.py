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
import train_tf as base_train
from pose.config.config import config
from pose.config.config import update_config
from pose.tf_core.tf_function_kd import train, validate
import tensorflow as tf
from pose.tf_core import tf_loss
from easydict import EasyDict as edict
from pose.utils import torch_tools


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


class Trainer(base_train.Trainer):
    def __init__(self, args):
        self.args = args
        self.gpus = [int(i) for i in args.gpu_id.split(',')]
        self.device = "cuda:{}".format(self.gpus[0])
        print("gpus:{}".format(self.gpus))
        super(Trainer, self).__init__(args)
        self.args = args
        self.teacher_model1 = self.get_teadher_model1(self.device)
        self.teacher_model2 = self.get_teadher_model2(self.device)
        # self.teacher_model = {"pose_resnetst": self.teacher_model1, "pose_hrnet": self.teacher_model2}
        # self.teacher_model = {"pose_resnetst": self.teacher_model1}
        self.teacher_model = {"pose_hrnet": self.teacher_model2}
        self.kd_criterion = tf_loss.JointsMSE_Loss2
        # self.kd_weight = {"deconv_layers": 1.0, "final_layer": 1.0}
        self.kd_weight = {"final_layer": 1.0}
        self.logger.info("Teacher:{}".format(self.teacher_model.keys()))
        self.logger.info("kd_weight:{}".format(self.kd_weight))

    def get_teadher_model1(self, device="cuda:0"):
        """
        :return:
        """
        from pose.nets.build_nets import build_nets
        from configs import teacher_config

        t_config = teacher_config.custom_coco_finger4_model_pose_resnetst_256_256
        t_config = edict(t_config)
        if args.polyaxon:
            from pose.utils import rsync_data
            # transforms_v1
            # model_path = "dataset/dataset/models/finger4-transform/pose_resnetst_50_256x256_0.001_adam_finger_transforms_v1_2020-10-28-10-21/model/best_model_182_0.9860.pth"
            model_path = "dataset/dataset/models/finger4-transform/pose_resnetst_50_256x256_0.001_adam_gaussian_finger_transforms_v2_2020-11-03-10-15/model/best_model_174_0.9822.pth"
            t_config.TEST.model_file = rsync_data.get_polyaxon_dataroot(root="ceph", dir=model_path)
        # self.input_size = (256, 192)  # h,W
        net_type = t_config.MODEL.NAME
        model_path = t_config.TEST.model_file
        model = build_nets(net_type=net_type, config=t_config, is_train=False)
        state_dict = torch_tools.load_state_dict(model_path, module=False)
        model.load_state_dict(state_dict)
        model = model.eval()
        model = model.to(device)
        self.logger.info('=> load teacher_model1: {}'.format(model_path))
        return model

    def get_teadher_model2(self, device="cuda:0"):
        """
        :return:
        """
        from pose.nets.build_nets import build_nets
        from configs import teacher_config
        t_config = teacher_config.custom_coco_finger4_model_pose_hrnet_256_256
        t_config = edict(t_config)
        if args.polyaxon:
            from pose.utils import rsync_data
            # transforms_v1
            # model_path = "dataset/dataset/models/finger4-transform/pose_hrnet_48_256x256_0.001_adam_finger_transforms_v1_2020-10-29-17-20/model/model_pose_hrnet_192_0.9882.pth"
            model_path = "dataset/dataset/models/finger4-transform/pose_hrnet_48_256x256_0.001_adam_gaussian_finger_transforms_v2_2020-11-03-10-28/model/model_pose_hrnet_195_0.9881.pth"
            t_config.TEST.model_file = rsync_data.get_polyaxon_dataroot(root="ceph", dir=model_path)
        # self.input_size = (256, 192)  # h,W
        net_type = t_config.MODEL.NAME
        model_path = t_config.TEST.model_file
        model = build_nets(net_type=net_type, config=t_config, is_train=False)
        state_dict = torch_tools.load_state_dict(model_path, module=False)
        model.load_state_dict(state_dict)
        model = model.eval()
        model = model.to(device)
        self.logger.info('=> load teacher_model2: {}'.format(model_path))
        return model

    def get_teadher_model_tf(self, device="cuda:0"):
        """
        :return:
        """
        from pose.nets.tf_build_nets import build_nets
        from configs import tf_val_config as val_config
        t_config = val_config.custom_coco_finger4_model_pose_resnetst_256_256
        t_config = edict(t_config)
        # t_config.TEST.MODEL_FILE = '/home/data/dataset3/models/finger-keypoints/finger4-v1/pose_resnetst_50_256x256_0.001_adam_finger_pretrained_2020-10-22-11-38/model/best_model_174_0.9853.pth'
        # self.input_size = (256, 192)  # h,W
        net_type = t_config.MODEL.NAME
        model_path = t_config.TEST.model_file
        model = build_nets(net_type=net_type, config=t_config, is_train=False)
        state_dict = torch_tools.load_state_dict(model_path, module=False)
        model.load_state_dict(state_dict)
        model = model.eval()
        model = model.to(device)
        self.logger.info('=> load teacher_model: {}'.format(model_path))
        return model

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
            train(config, self.train_loader, self.model, self.teacher_model, self.criterion, self.kd_criterion,self.kd_weight,
                  self.optimizer, epoch, self.work_dir, self.tb_log_dir, self.writer_dict, device=self.device)
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


if __name__ == '__main__':
    args = parse_args()
    t = Trainer(args)
    t.start_train()
