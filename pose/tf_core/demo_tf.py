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
import torch
import numpy as np

sys.path.append(os.path.dirname(__file__))
from utils import image_utils, debug, file_utils
from easydict import EasyDict as edict
from libs.detector import detect_human
from pose.core.inference import get_final_preds
from pose.nets.tf_build_nets import build_nets
from utils import tf_tools
from configs import tf_val_config as val_config
from pose.tf_core import tflite_detector, tfpb_detector, tf_detector, onnx_detector
import demo
import tensorflow as tf

project_root = os.path.dirname(__file__)


class PoseEstimation(demo.PoseEstimation):
    """
     mpii_keypoints_v2 = {0: "r_ankle", 1: "r_knee", 2: "r_hip", 3: "l_hip", 4: "l_knee", 5: "l_ankle", 6: "pelvis",
                         7: "thorax", 8: "upper_neck", 9: "head_top", 10: " r_wrist", 11: "r_elbow", 12: "r_shoulder",
                         13: "l_shoulder", 14: "l_elbow", 15: "l_wrist"}

    mpii_keypoints = {"r_ankle": 0, "r_knee": 1, "r_hip": 2, "l_hip": 3, "l_knee": 4, "l_ankle": 5, "pelvis": 6,
                      "thorax": 7, "upper_neck": 8, "head_top": 9, " r_wrist": 10, "r_elbow": 11, "r_shoulder": 12,
                      "l_shoulder": 13, "l_elbow": 14, "l_wrist": 15}
    """

    def __init__(self, config, threshold=0.85, device="cuda:0"):
        """
        :param config:
        :param threshold:
        :param device:
        """
        self.config = edict(config)
        print(self.config)
        self.threshold = threshold
        self.device = device
        # coco_skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
        #                  [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
        # self.skeleton = coco_skeleton
        self.skeleton = self.config.TEST.skeleton
        # self.skeleton = custom_mpii_skeleton

        # self.input_size = (256, 192)  # h,W
        self.input_size = tuple(self.config.MODEL.IMAGE_SIZE)  # h,w
        self.net_type = self.config.MODEL.NAME
        self.model_path = self.config.TEST.model_file
        self.transform = self.get_transforms()
        # init SSDDetector
        self.detector = detect_human.HumanDetection(detector_type="SSD", device=device)
        self.model_path = os.path.join(project_root, self.model_path)
        self.model = self.build_model(self.net_type, self.model_path)

    def build_model(self, net_type, model_path):
        """
        build model
        :param net_type:
        :param model_path:
        :return:
        """
        if "tflite" in model_path:
            model = tflite_detector.TFliteDetector(tflite_path=model_path)
        elif "pb" in model_path:
            model = tfpb_detector.TFPBDetector(pb_path=model_path)
        elif "onnx" in model_path:
            model = onnx_detector.ONNXModel(onnx_path=model_path)
        elif "h5" in model_path or os.path.exists(os.path.join(model_path, "saved_model.pb")):
            model = tf_detector.TFDetector(model_path=model_path)
        else:
            model = build_nets(net_type=net_type, config=self.config, is_train=False)
            model.load_weights(self.model_path)
            # model.load_weights(self.model_path, by_name=False, skip_mismatch=True)
        print("load model file from:{}".format(model_path))
        return model

    @debug.run_time_decorator("forward")
    def forward(self, input_tensor):
        output_tensor = self.model(input_tensor)
        return output_tensor

    def start_capture(self, video_path, save_video=None, detect_freq=1):
        """
        start capture video
        :param video_path: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        video_cap = image_utils.get_video_capture(video_path)
        width, height, numFrames, fps = image_utils.get_video_info(video_cap)
        if save_video:
            self.video_writer = image_utils.get_video_writer(save_video, width, height, fps)
        # freq = int(fps / detect_freq)
        count = 0
        while True:
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            if count % detect_freq == 0:
                key_points, kp_scores, body_rects = self.detect_image(frame,
                                                                      threshold=self.threshold,
                                                                      use_box=False)
                frame = self.draw_result(frame, key_points, kp_scores, body_rects, delay=10)
            if save_video:
                self.video_writer.write(frame)
            count += 1
        video_cap.release()

    @debug.run_time_decorator("detect")
    def detect(self, image, boxes, threshold):
        if len(boxes) > 0:
            body_rects = image_utils.bboxes2rects(boxes)
            key_points = []
            kp_scores = []
            for body_rect in body_rects:
                points, kp_score = self.inference(image, body_rect, threshold=threshold)
                key_points.append(points)
                kp_scores.append(kp_score)
        else:
            key_points, kp_scores, body_rects = [], [], []
        # return key_points
        return key_points, kp_scores, body_rects

    @debug.run_time_decorator("detect_image")
    def detect_image(self, frame, threshold=0.8, use_box=False):
        '''
        :param frame: bgr image
        :param threshold:
        :return:
        '''
        if use_box:
            boxes, scores = self.detect_targets(frame)
        else:
            h, w, d = frame.shape
            boxes = [[0, 0, w, h]]
        key_points, kp_scores, body_rects = self.detect(frame, boxes, threshold)
        return key_points, kp_scores, body_rects

    def inference(self, bgr_image, body_rect, threshold=0.1):
        """
        :param bgr_image:
        :param body_rect:
        :param threshold:
        :return:
        """
        # image, center, scale = self.pre_process3(bgr_image, body_rect)
        image, center, scale = self.pre_process2(bgr_image, body_rect)
        # image_processing.show_image_rects("body_rect", bgr_image, [body_rect])
        # image_processing.cv_show_image("image", image, waitKey=0)
        input_tensor = self.transform(image)
        # input_tensor = image_processing.image_normalization(image,
        #                                                     mean=[0.485, 0.456, 0.406],
        #                                                     std=[0.229, 0.224, 0.225])
        # input_tensor = image_processing.image_normalization(image, mean=[127., 127., 127.], std=[127., 127., 127.])
        # input_tensor = image
        # input_tensor = input_tensor.transpose(2, 0, 1)  # [H0,W1,C2]-[C,H,W]
        # input_tensor = torch.from_numpy(input_tensor)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.numpy().transpose(0, 2, 3, 1)  # <class 'tuple'>: (16, 256, 192, 3)

        output = self.forward(input_tensor)
        output = output.transpose(0, 3, 1, 2)
        # print("output:{}".format(output[0, 0, 0, :]))
        key_point, kp_score = self.post_process(bgr_image, output, center, scale, threshold)
        return key_point, kp_score


if __name__ == '__main__':
    # hp = PoseEstimation(config=val_config.coco_res50_192_256, device="cuda:0")
    # hp = PoseEstimation(config=val_config.coco_res18_192_256, device="cuda:0")
    # hp = PoseEstimation(config=val_config.mpii_256_256, device="cuda:0")
    # hp = PoseEstimation(config=val_config.custom_coco_finger_res18_192_256, device="cuda:0")
    # hp = PoseEstimation(config=val_config.custom_coco_finger4_model_mbv2_192_256, device="cuda:0")
    hp = PoseEstimation(config=val_config.custom_coco_finger_res_256_256, device="cuda:0")
    # hp = PoseEstimation(config=val_config.custom_coco_finger_model_mbv2_192_256, device="cuda:0")
    # hp = PoseEstimation(config=val_config.custom_coco_person_res18_192_256, device="cuda:0")
    # hp = PoseEstimation(config=val_config.custom_mpii_256_256, device="cuda:0")
    # hp = PoseEstimation(config=val_config.student_mpii_256_256, device="cuda:0")
    # hp = PoseEstimation(config=val_config.student_mpii_256_256_v2, device="cuda:0")
    # video_path = "/home/data/dataset3/dataset/finger/finger-detection/VID_20200814_110845.mp4"
    # save_video = "/home/data/dataset3/dataset/finger/finger-detection/VID_20200814_110845_keypoiny4_kd_tf.avi"
    video_path = "/home/data/dataset3/git_project/灵犀指 AIP/video/click1.3gp"
    save_video = "/home/data/dataset3/git_project/灵犀指 AIP/video/click1_kp4.avi"
    hp.start_capture(video_path=video_path, save_video=save_video)
    # hp.start_capture(video_path)
    # run_time_test()
    image_dir = "/home/data/dataset3/release/AIT/finger-keypoint-detection/data/test_image"
    # image_dir = "/home/data/dataset3/dataset/finger_keypoint/finger/val.txt"
    # image_dir = "/home/data/dataset3/dataset/finger/Bug/JPEGImages"
    # hp.batch_test_image(image_dir, use_box=True, waitKey=0)
    hp.detect_image_dir(image_dir, use_box=False, delay=0)
    # hp.detect_image_val(image_dir, use_box=False, waitKey=0)
