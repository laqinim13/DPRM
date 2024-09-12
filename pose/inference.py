# -*-coding: utf-8 -*-
"""
    @Project: torch-Human-Pose-Estimation-Pipeline
    @File   : demo.py
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-11-08 15:02:19
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))
import numpy as np
import cv2
import copy
import torch
import torchvision.transforms as transforms
from typing import List
from easydict import EasyDict as edict
from pose.utils.transforms import get_affine_transform
from pose.core.inference import get_final_preds, get_final_preds_offset
from pose.nets.build_nets import build_nets
from pose.utils import setup_config, torch_tools
from pybaseutils import image_utils, json_utils
from pybaseutils.pose import bones_utils
from pose.utils import my_bones_utils
from pose.utils import os_image_utils

pwd = os.path.dirname(__file__)


class PoseEstimation(object):
    def __init__(self, config_file, model_file=None, target: str = None, threshold=0.3, device="cuda:0"):
        """
        :param config_file: 配置文件
        :param model_file: 模型文件
        :param target: 关键点类别: hand,coco_person,mpii
        :param threshold: 阈值
        :param device: 运行设备
        """
        self.config = setup_config.load_config(config_file)
        self.config = setup_config.parser_config_file(self.config, os.path.join(pwd, "config/default.yaml"), True)
        self.config.TEST.POST_PROCESS = True
        self.threshold = threshold
        self.device = device
        target_bones = my_bones_utils.get_target_bones(target)
        if target_bones:
            self.skeleton = target_bones["skeleton"]
            self.colors = target_bones["colors"]
        else:
            self.skeleton = json_utils.get_value(self.config, ["DATASET", "SKELETON"], [])
            self.colors = None
        self.input_size = tuple(self.config.MODEL.IMAGE_SIZE)  # w,h
        self.net_type = self.config.MODEL.NAME
        if not model_file:
            model_file = self.config.TEST.MODEL_FILE
        self.model_path = model_file
        self.transform = self.get_transforms()
        # self.model_path = os.path.join(project_root, self.model_path)
        self.model = self.build_model(self.net_type, self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.scale_rate = self.config.DATASET.SCALE_RATE
        # self.scale_rate = 1.25

    def get_transforms(self):
        """
        input_tensor = image_processing.image_normalization(image,
                                                            mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.transpose(2, 0, 1)  # [H0,W1,C2]-[C,H,W]
        input_tensor = torch.from_numpy(input_tensor)
        :return:
        """
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # b,g,r
        #                          std=[0.229, 0.224, 0.225]),
        # ])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform

    def build_model(self, net_type, model_path):
        """
        build model
        :param net_type:
        :param model_path:
        :return:
        """
        model = build_nets(net_type=net_type, config=self.config, is_train=False)
        state_dict = torch_tools.load_state_dict(model_path)
        model.load_state_dict(state_dict)
        return model

    def detect(self, bgr, boxes, threshold=0.3):
        """
        :param bgr: bgr image
        :param boxes:  [xmin, ymin, xmax, ymax]
        """
        kp_points, kp_scores = [], []
        for box in boxes:
            points, scores = self.inference(bgr, box, threshold)
            kp_points.append(points)
            kp_scores.append(scores)
        return kp_points, kp_scores

    def pre_process(self, image, box):
        input, center, scale = self.get_input_center_scale(image, box)
        input_tensor = self.transform(input)
        input_tensor = torch.unsqueeze(input_tensor, 0)
        input_tensor = input_tensor.to(self.device)
        # image_processing.show_image_rects("body_rect", bgr_image, [body_rect])
        # image_processing.cv_show_image("image", image, waitKey=0)
        # input_tensor = self.transform(image)
        # input_tensor = self.get_transform(image)
        # input_tensor = np.asarray(image / 255.0, dtype=np.float32)
        # input_tensor = image_processing.image_normalization(image,
        #                                                     mean=[0.485, 0.456, 0.406],
        #                                                     std=[0.229, 0.224, 0.225])
        # input_tensor = input_tensor.transpose(2, 0, 1)  # [H0,W1,C2]-[C,H,W]
        # input_tensor = torch.from_numpy(input_tensor)
        # input_tensor = input_tensor.unsqueeze(0)
        return input_tensor, center, scale

    def inference(self, bgr, box, threshold=0.1):
        """
        input_tensor = image_processing.image_normalization(image,
                                                             mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.transpose(2, 0, 1)  # [H0,W1,C2]-[C,H,W]
        input_tensor = torch.from_numpy(input_tensor)
        :param bgr:
        :param box:
        :param threshold:
        :return:
        """
        input_tensor, center, scale = self.pre_process(bgr, box)
        output = self.forward(input_tensor).cpu().numpy()
        kp_point, kp_score = self.post_process(output, center, scale, threshold)
        return kp_point, kp_score

    def forward(self, input_tensor):
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output_tensor = self.model(input_tensor)
        return output_tensor

    def __get_crop_images(self, image, box):
        crop_image = os_image_utils.get_bboxes_image(image, bboxes_list=[box])[0]
        crop_image = os_image_utils.resize_image(crop_image, size=(self.input_size[0], None))
        crop_image = os_image_utils.center_crop_padding(crop_image, crop_size=self.input_size)
        return crop_image

    def get_input_center_scale(self, image, box):
        '''
        :param image: 图像
        :param box: 检测框
        :return: 截取的当前检测框图像，中心坐标及尺寸
        '''
        aspect_ratio = 0.75
        pixel_std = 200
        scale_rate = self.scale_rate

        def _box2cs(box):
            x = box[0]
            y = box[1]
            w = box[2] - box[0]
            h = box[3] - box[1]
            return _xywh2cs(x, y, w, h)

        def _xywh2cs(x, y, w, h):
            center = np.zeros((2), dtype=np.float32)
            center[0] = x + w * 0.5
            center[1] = y + h * 0.5

            if w > aspect_ratio * h:
                h = w * 1.0 / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio
            scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
            if center[0] != -1:
                # scale = scale * 1.25
                scale = scale * scale_rate
            return center, scale

        box = copy.deepcopy(box)
        center, scale = _box2cs(box)
        trans = get_affine_transform(center, scale, 0, self.input_size)
        input = cv2.warpAffine(image, trans, (self.input_size[0], self.input_size[1]), flags=cv2.INTER_LINEAR)
        return input, center, scale

    def post_process(self, heatmap, center, scale, threshold):
        # compute coordinate
        kp_point, kp_score = self.get_final_output(heatmap, center, scale, threshold)
        # kp_point, kp_score = get_final_preds(self.config, heatmap.clone().cpu().numpy(), np.asarray([center]),
        #                                    np.asarray([scale]))
        return kp_point, kp_score

    def get_final_output(self, pred, center, scale, threshold=0.0):
        # compute coordinate
        if self.config.MODEL.OUT_TYPE == 'gaussian':
            kp_point, kp_score = get_final_preds(self.config, pred, np.asarray([center]), np.asarray([scale]))
        else:
            self.config.LOSS = edict()
            self.config.LOSS.KPD = 4.0
            kp_point, kp_score, _ = get_final_preds_offset(self.config, pred, np.asarray([center]),
                                                           np.asarray([scale]))
        kp_point, kp_score = kp_point[0, :], kp_score[0, :]
        # for custom_mpii_256_256 cal head coordinate
        # kp_point[3, :] = (kp_point[3, :] + kp_point[2, :]) / 2
        # score[3] = (score[3] + score[2]) / 2
        index = kp_score < threshold
        index = index.reshape(-1)
        kp_point[index, :] = (0, 0)
        kp_point = np.abs(kp_point)
        return kp_point, kp_score

    @staticmethod
    def center_scale2rect(center, scale, pixel_std=200):
        w = pixel_std * scale[0]
        h = pixel_std * scale[1]
        x = center[0] - 0.5 * w
        y = center[1] - 0.5 * h
        rect = [x, y, w, h]
        return rect

    @staticmethod
    def adjust_center_scale(center, scale, alpha=15.0, beta=1.25, type="center_default"):
        '''
         Adjust center/scale slightly to avoid cropping limbs
        if  c[0] != -1:
            c[1] = c[1] + 15 * s[1]
            s = s * 1.25
        :param center:
        :param scale:官方的说法是：person scale w.r.t. 200 px height,理解是这个scale=图片中人体框的高度/200
        :param alpha:
        :param beta:
        :return:
        '''
        if center[0] != -1:
            if type == "center_up":
                rect = PoseEstimation.center_scale2rect(center, scale)
                x, y, w, h = rect
                center[0] = x + w * 0.5
                center[1] = y + h * 0.5 + alpha * h
                scale = scale * beta
            elif type == "center_default":
                center[1] = center[1] + alpha * scale[1]
                scale = scale * beta
            elif type == "center":
                center = center
                scale = scale
        return center, scale

def draw_result(image, boxes, points, pada_string, delay=0):
    """
    :param image:
    :param boxes: 检测框
    :param points: 关键点
    :param scores: 关键点置信度
    :param skeleton:  关键点连接顺序
    :param delay:
    """
    # 如果关键点为空，则创建一个空列表kpts
    # kpts = [] if len(points) == 0 else np.concatenate([np.array(points), np.array(scores)], axis=2)
    # print(kpts) #输出点
    # 使用image_utils工具在图像上绘制关键点和边界框
    image = os_image_utils.draw_key_point_in_image(image, points,pada_string=pada_string, pointline=[],
                                                boxes=boxes, thickness=1)
    # 使用image_utils工具显示图像
    # os_image_utils.cv_show_image('frame', image, use_rgb=False, delay=delay)
    # 返回绘制后的图像
    return image
