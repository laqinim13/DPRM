# -*-coding: utf-8 -*-
"""
    @File   : demo.py
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2019-11-08 15:02:19
"""
import sys
import os
import torch
import wave
import contextlib
from collections import Counter

# sys.path.append("libs/detector/libs/detector")
# sys.path.append(os.path.dirname(__file__))
import cv2
import numpy as np
import argparse
from easydict import EasyDict
# from libs.detector.libs.detector.detector import Detector
from pybaseutils import image_utils, file_utils
from pose.smooth import pose_filter
from pose.inference import draw_result as ps
# from pose import inference_duttar
from yolov5.re_project.detector import Detector
from pose.utils import duttar_point_hui_gui
from pose.utils.hand_duttar_utils.hand_pos import Hand_Pose
from pose.utils.hand_duttar_utils.duttar_pos import Duttar_Pose
from pose.utils.hand_duttar_utils.saoxian_pos import Saoxian_Pose
from pose.utils.hand_duttar_utils.pada import pada
# from conformer import infer_path
import subprocess



project_root = os.path.dirname(__file__)

# 总扫弦时间
saoxian_time = []
# 总扫弦状态
saoxian_text = []
# 最后的pada 对连续相同的品位单独（没有扫弦状态） 合并
pada_last = []
# 最后的pada 对连续相同的品位单独（没有扫弦状态） 合并 的时间
pada_time_last = []
# 最后的pada和time 合并 （应该没用）
pada_time_together = []
# 最后的输出列表品位（不是方案里的）
output_list = []
# 品位 2
pada_2 = ["2"]
# pada的索引
time_pada = 1
# 扫弦对于品位时间所对应的索引
pada_position = []
# 品位位置（不连续的去掉）
print_pada = []
# 方案1
result_pada = []
# 方案2
result_pada_2 = []

class Pose():
    def __init__(self,
                 # config_file,
                 # config_file1,
                 # model_file=None,
                 # model_file1=None,
                 # target: str = "",
                 # target1: str = "",
                 use_box=True,
                 threshold=0.5,
                 filter_id=[],
                 device="cuda:0"):
        """
        :param config_file: 配置文件
        :param model_file: 模型文件
        :param target: 关键点类别 hand,coco_person,mpii
        :param use_box: 使用检测框
        :param threshold: 阈值
        :param device: 运行设备
        """
        # super(PoseEstimation, self).__init__(config_file, model_file, target, threshold, device)
        # super(PoseEstimation, self).__init__(config_file1, model_file1, target1, threshold, device)
        self.threshold = threshold
        self.use_box = use_box
        self.hand = Hand_Pose(device=device)
        self.duttar = Duttar_Pose(device=device)
        self.saoxian = Saoxian_Pose()
        self.detecter = Detector(device=device)
        self.filter_id = filter_id
        if self.filter_id:
            self.pose_filter = pose_filter.PoseFilter(filter_id=self.filter_id, win_size=5, decay=0.5)

    def sao_xian(self,real_time_demo=True,use_pun=False,is_itn=False):
        time_list, text = self.saoxian.detect(real_time_demo=True,use_pun=False,is_itn=False)
        global saoxian_time
        global saoxian_text
        saoxian_time = time_list
        saoxian_text = text
        # print(time_list)
        # print()
        # print(text)

    def start_capture(self, video_file, save_video=None, use_box=False, interval=1):
        """
        start capture video
        :param video_file: *.avi,*.mp4,...
        :param save_video: *.avi
        :param use_box:
        :param detect_freq:
        :return:
        """
        # infer_path.real_time_predict_demo()
        video_cap = image_utils.get_video_capture(video_file)
        width, height, num_frames, fps = image_utils.get_video_info(video_cap)
        video_writer = None
        # fps = max((fps + interval - 1) // interval, 2)
        fps = max(fps // interval, 2)
        if save_video: video_writer = image_utils.get_video_writer(save_video, width, height, fps)
        # freq = int(fps / detect_freq)
        count = 0
        global pada_time_last
        global time_pada
        global pada_position
        global result_pada_2
        while True:
            if count % interval == 0:
                # 设置抽帧的位置
                if isinstance(video_file, str): video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                isSuccess, frame = video_cap.read()
                if not isSuccess or 0 < num_frames < count: break
                pada_time_last.append(round((count/fps), 2))
                print(pada_time_last)
                print(len(pada_time_last))
                print(pada_time_last[-1])
                print(time_pada)
                if pada_time_last[-1] <= saoxian_time[time_pada]:
                    # print(saoxian_time[time_pada])
                    # print(saoxian_text[time_pada-1])
                    saoxian_dang_qian = saoxian_text[time_pada-1]
                    if saoxian_dang_qian == "上" :
                        saoxian_dang_qian = "s"
                    else:
                        saoxian_dang_qian = "x"
                else:
                    time_pada += 1
                    pada_position.append(len(pada_time_last) - 1)   # 将添加到列表末尾
                    # pada_position += (len(pada_time_last) - 1)
                    # print(pada_position)
                points, scores, boxes, skeleton, pada_string = self.detect_image(frame, self.threshold, use_box=self.use_box)
                pada_string = pada_string + "  " + saoxian_dang_qian
                result_pada_2.append(pada_string)
                frame = ps(image=frame, boxes=boxes, points=points,pada_string=pada_string, delay=5)
                # frame = ps.draw_result(frame, boxes, points, scores=[], skeleton=[], delay=5)
                if save_video:
                    video_writer.write(frame)
            count += 1
        video_cap.release()

    def detect_targets(self, bgr):
        """目标检测"""
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bbox_score = self.detecter.detect(image)
        return bbox_score

    def detect_pose(self, bgr, boxes, threshold, number):
        if number == 1:
            """检测Keypoint"""
            points, scores, skeleton = self.hand.detect(bgr, boxes, threshold=threshold)
            if self.filter_id:
                self.pose_filter.filter(points[0])
            return points, scores, skeleton
        elif number == 2:
            """检测Keypoint"""
            points, scores, skeleton = self.duttar.detect(bgr, boxes, threshold=threshold)
            if self.filter_id:
                self.pose_filter.filter(points[0])
            return points, scores, skeleton

    def detect_image(self, frame, threshold=0.8, use_box=False):
        """
        :param frame: bgr image
        :param threshold:
        :param use_box: 是否检测人体
        :return:
        """
        global pada_last
        global output_list
        global pada_time_together
        global print_pada
        # dets = []
        if use_box:
            dets = self.detect_targets(frame)
        # 如果不使用边界框，则默认整个图像为一个边界框
        else:
            h, w = frame.shape[:2]
            boxes = [[0, 0, w, h]]
        # boxes = dets[:, 0:4]
        # score_ = dets[:, 4:5]
        # labels = dets[:, 5:6]
        # print(dets)
        # 调用detect_pose方法来检测姿态
        # 循环遍历每一个元素
        new_han = []
        new_duttar = []
        # hand_points = []
        # hand_scores = []
        # hand_skeleton = []
        #
        # duttar_points = []
        # duttar_scores = []
        # duttar_skeleton = []
        for row in dets:
            if row[-1] == 0:
                number = 1
                new_han.append(row[:4])
                new_han = np.array(new_han)
                hand_points, hand_scores, hand_skeleton = self.detect_pose(frame, new_han, threshold, number)

            elif row[-1] == 2:
                number1 = 2
                new_duttar.append(row[:4])
                new_duttar = np.array(new_duttar)
                duttar_points, duttar_scores, duttar_skeleton = self.detect_pose(frame, new_duttar, threshold, number1)
                duttar_points, m, b = duttar_point_hui_gui.duttar_point_hui_gui(duttar_points)

        pada_returen = pada(hand_points, duttar_points, m, b)
        print_pada_1 = []
        if len(pada_returen) != 0:
            print_pada.append(pada_returen)
            print_pada_1.append(pada_returen)
            pada_time_together.append(pada_returen)
            if len(pada_last) != 0:
                if pada_returen[0] != pada_last[-1]:
                    pada_last.append(pada_returen)
            else:
                pada_last.append(pada_returen)
        else:
            pada_time_together.append(pada_2)
            print_pada.append(pada_2)
            print_pada_1.append(pada_2)
            pada_last.append(pada_2)
        # print(print_pada_1[0][0])
        # 品位输出在画面的变量
        pada_string = print_pada_1[0][0]
        # print(pada_time_together)
        # print(len(pada_time_together))
        # 最后的结果，去掉了重复的
        output_list = []
        prev_char = None
        for char in pada_last:
            if char != prev_char:
                output_list.append(char)
            prev_char = char
        # print(output_list)
        # 创建一个新列表，用于存放结果
        # points = []
        # # 将list1和list2的元素添加到combined_list中
        # points.extend(hand_points)
        # points.extend(duttar_points)
        #
        # scores = []
        # # 将list1和list2的元素添加到combined_list中
        # scores.extend(hand_scores)
        # scores.extend(duttar_scores)
        # print("hand_points")
        # print(hand_points)
        # print("duttar_points")
        # print(duttar_points)

        points = hand_points + duttar_points
        scores = hand_scores + duttar_scores
        new_tensor =np.vstack((new_han, new_duttar))
        skeleton = hand_skeleton + duttar_skeleton
        # print(new_tensor, labels)
        # 返回检测到的关键点坐标、置信度和边界框坐标
        return points, scores, new_tensor,skeleton, pada_string

    def detect_image_dir(self, image_dir, out_dir=None, delay=0):
        """
        :param image_dir: image directory or image file ,txt file
        :param out_dir: save result image directory
        :param delay:
        """
        image_list = file_utils.get_files_lists(image_dir, shuffle=False)
        for i, image_file in enumerate(image_list):
            # print(image_file)
            image = cv2.imread(image_file)
            image = image_utils.resize_image(image, size=(640, None))
            points, scores, boxes,skeleton, pada_string = self.detect_image(image, threshold=self.threshold, use_box=self.use_box)

            image = ps(image, boxes, points, pada_string, delay=5)
            if out_dir:
                save_file = file_utils.create_dir(out_dir, None, os.path.basename(image_file))
                cv2.imwrite(save_file, image)


def demo_coco_hand(cfg: EasyDict = EasyDict()):
    # cfg.video_file = 1
    cfg.video_file = "my_test/1.mp4"
    cfg.out_dir = "output"
    return cfg

def get_parser():
    cfg = EasyDict()
    # cfg.config_file = None
    # cfg.model_file = None
    # cfg.config_file1 = None
    # cfg.model_file1 = None

    # cfg.target = None
    # cfg.target1 = None

    cfg.image_dir = None
    cfg.out_dir = None
    cfg.video_file = None
    cfg.threshold = 0.2
    cfg = demo_coco_hand(cfg)
    parser = argparse.ArgumentParser(description="Demo Test")

    parser.add_argument("--image_dir", help="image file or directory", default=cfg.image_dir, type=str)
    parser.add_argument("--out_dir", help="save result directory", default=cfg.out_dir, type=str)
    parser.add_argument("--video_file", help="input video file", default=cfg.video_file, type=str)
    parser.add_argument("--threshold", help="threshold", default=cfg.threshold, type=float)
    parser.add_argument("--device", help="GPU device id", default="cuda:0", type=str)
    return parser

def vidio2wav(input_video, output_audio="my_test/test_wav.wav", sample_rate=16000):
    try:
        # 使用FFmpeg提取音频并重新采样
        subprocess.run(['ffmpeg', '-i', input_video, '-vn', '-ar', str(sample_rate), output_audio], check=True)
        print("音频视频已成功分离！")
    except subprocess.CalledProcessError as e:
        print("提取音频时出错:", e)

def merge_video_audio(video_file="output/linux_1.avi", audio_file="my_test/test_wav.wav", output_file="output/output_video.mp4"):
    try:
        # 使用FFmpeg合并视频和音频
        subprocess.run(['ffmpeg', '-i', video_file, '-i', audio_file, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_file], check=True)
        print("视频和音频合并成功！")
    except subprocess.CalledProcessError as e:
        print("合并视频和音频时出错:", e)
def wav_len_1(audio_file="my_test/test_wav.wav"):
    with contextlib.closing(wave.open(audio_file, 'r')) as wav_file:
        # 获取音频文件的帧数（frames）和帧率（frame rate）
        num_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()

        # 计算音频的总时长（秒）
        audio_length = num_frames / frame_rate

        # 保留小数点后三位，格式化输出
        audio_length_formatted = "{:.3f}".format(audio_length)
    return audio_length_formatted

def zui_hou_jie_guo():
    global result_pada
    b = 0
    for i in range(len(pada_position)):
        start_index = b
        # print(start_index)
        b = pada_position[i]
        end_index = b
        # print(end_index)
        # 切片区域
        interest_region = print_pada[start_index:end_index]
        # print(interest_region)
        # 将列表扁平化为一个包含单个元素的列表
        flat_list = [item for sublist in interest_region for item in sublist]
        # 使用Counter统计元素出现的次数
        # 使用 Counter 来计数
        counter = Counter(flat_list)
        # 找出出现次数最多的元素
        most_common_element = counter.most_common(1)[0][0]
        count = counter.most_common(1)[0][1]
        result_pada.append(most_common_element)  # 将添加到列表末尾
        # print(f"最多出现的元素：{most_common_element},出现次数{count}")
        # print(result_pada)
        # print(b)
        # print(result_pada)

def zui_hou_jie_guo_2():
    result_list = []
    result_str = []
    result_time = []

    consecutive_count = 1
    current_index = 0
    for i in range(1, len(result_pada_2)):
        if result_pada_2[i] == result_pada_2[i - 1]:
            consecutive_count += 1
        else:
            if consecutive_count >= 2:
                result_list.append(current_index + consecutive_count - 1)
                result_str.append(result_pada_2[current_index + consecutive_count - 1])
                result_time.append(pada_time_last[current_index + consecutive_count - 1])

            consecutive_count = 1
            current_index = i

    if consecutive_count >= 2:
        result_list.append(current_index + consecutive_count - 1)
        result_str.append(result_pada_2[current_index + consecutive_count - 1])
        result_time.append(pada_time_last[current_index + consecutive_count - 1])

    return result_list, result_str, result_time

if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()
    vidio2wav(input_video=opt.video_file)
    pose = Pose(
                threshold=opt.threshold,
                device=opt.device
                 )
    wav_len_2 = wav_len_1()
    pose.sao_xian(real_time_demo=True,use_pun=False,is_itn=False)
    # print(wav_len_2)
    # print(type(wav_len_2))
    # wav_len_2 = str(wav_len_2)
    wav_len_2 = float(wav_len_2)
    # print(type(wav_len_2))
    # print()
    # print(saoxian_time[-1])
    # print(type(saoxian_time[-1]))
    saoxian_time[-1] = wav_len_2 + 0.05
    # print(saoxian_time[-1])
    # print(saoxian_time)
    # print(saoxian_text)
    # print(len(saoxian_text))
    if isinstance(opt.video_file, str) or isinstance(opt.video_file, int):
        opt.video_file = str(opt.video_file)
        if len(opt.video_file) == 1: opt.video_file = int(opt.video_file)
        save_video = os.path.join(opt.out_dir, "linux_1.avi") if opt.out_dir else None
        pose.start_capture(video_file=opt.video_file, save_video=save_video)

    else:
        pose.detect_image_dir(image_dir=opt.image_dir, out_dir=opt.out_dir)
    merge_video_audio()
    print("1.品位状态：")
    print("品位时间：")
    print(pada_time_last)
    print(len(pada_time_last))
    print("品位位置：")
    print(output_list)
    print(len(output_list))
    print("品位位置（不连续的去掉）：")
    print(print_pada)
    print(len(print_pada))
    print("2.扫弦状态：")
    print("扫弦时间")
    print(saoxian_time)
    print("扫弦状态")
    print(saoxian_text)
    print(len(saoxian_text))
    print("根据扫弦对应品位时间索引")
    pada_position.append(len(pada_time_last) - 1)  # 将添加到列表末尾
    print(pada_position)
    print(len(pada_position))
    print("最后的结果：")
    zui_hou_jie_guo()
    print("方案1：")
    print("品位：")
    print(result_pada)
    print(len(result_pada))
    print("扫弦：")
    print(saoxian_text)
    print(len(saoxian_text))
    print("方案2：")
    print("品位：")
    print("扫弦和品位合为一个列表")
    print(result_pada_2)
    print(len(result_pada_2))
    print("根据扫弦和品位合为一个列表留下连续相同的")
    result_list, result_str, result_time = zui_hou_jie_guo_2()
    print("相同最后的索引")
    print(result_list)
    print(len(result_list))
    print("相同最后的品位和扫弦")
    print(result_str)
    print(len(result_str))
    print("每个一个品位和扫弦所对应的时间")
    print(result_time)
    print(len(result_time))


