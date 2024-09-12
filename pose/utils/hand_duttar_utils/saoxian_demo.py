import sys
import os
from pose import inference
from pose.smooth import pose_filter
import argparse
import functools
import time
import wave
# from masr.predict import MASRPredictor
from conformer.masr import predict
from conformer.masr.utils.utils import add_arguments, print_arguments

time_list = []
len_1 = 0

class MASRPredictor(predict.MASRPredictor):
    def __init__(self,
                 configs="conformer/configs/conformer.yml",
                 model_path="conformer/models1/conformer_streaming_fbank/inference.pt",
                 use_gpu=True,
                 pun_model_dir="conformer/models/pun_models/",
                 use_pun=False,
                 # wav_path="conformer/dataset/audio/data_aishell/wav/train/3/75.wav",
                 wav_path="my_test/test_wav.wav",
                 is_itn=False
                 ):
        super(MASRPredictor, self).__init__(configs="conformer/configs/conformer.yml", model_path="conformer/models1/conformer_streaming_fbank/inference.pt", use_gpu=True, pun_model_dir="conformer/models/pun_models/", use_pun=False)
        self.wav_path = wav_path
        self.is_itn = is_itn

    # 短语音识别
    def predict_audio(self, wav_path, use_pun=False, is_itn=False):
        start = time.time()
        result = self.predict(audio_data=wav_path, use_pun=use_pun, is_itn=is_itn)
        score, text = result['score'], result['text']
        print(f"消耗时间：{int(round((time.time() - start) * 1000))}ms, 识别结果: {text}, 得分: {int(score)}")
        return text

    # 长语音识别
    def predict_long_audio(self, wav_path, use_pun=False, is_itn=False):
        start = time.time()
        result = self.predict_long(audio_data=wav_path, use_pun=use_pun, is_itn=is_itn)
        score, text = result['score'], result['text']
        print(f"长语音识别结果，消耗时间：{int(round((time.time() - start) * 1000))}, 得分: {score}, 识别结果: {text}")
        return text

    # 实时识别模拟
    def real_time_predict_demo(self, wav_path, use_pun=False, is_itn=False):
        global time_list
        global len_1
        # 识别间隔时间
        interval_time = 0.05
        time_1 = 0
        time_list.append(time_1)
        CHUNK = int(16000 * interval_time)
        # 读取数据
        wf = wave.open(wav_path, 'rb')
        channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        data = wf.readframes(CHUNK)
        # 播放
        while data != b'':
            # time_list.append(time_1)
            # time_1 = time_1 + 0.05
            # print(time_list)
            start = time.time()
            d = wf.readframes(CHUNK)
            result = self.predict_stream(audio_data=data, use_pun=use_pun, is_itn=is_itn,
                                              is_end=d == b'',
                                              channels=channels, samp_width=samp_width, sample_rate=sample_rate)
            data = d
            if result is None:
                # time_list.append(time_1)
                time_1 = round(time_1 + 0.05, 3)
                # print(time_list)
                continue
            else:
                score, text = result['score'], result['text']
                text = text.replace("<unk>", "")
                a = len(text) - len_1
                len_1 = len(text)
                # print(
                #     f"【实时结果】：消耗时间：{int((time.time() - start) * 1000)}ms, 识别结果: {text}, 得分: {int(score)}")
                time_1 = round(time_1 + 0.05, 3)
                if a > 0:
                    end_element = time_list[-1]
                    qu_jian_zhi = time_1 - end_element
                    junzhi_qujian = qu_jian_zhi / a
                    for i in range(a):
                        time_list.append(round(end_element + junzhi_qujian * (i + 1), 3))
                # print(time_list)
        # 重置流式识别
        # print()
        # print(time_list)
        # print(text)
        # print(len(time_list))
        # print(len(text))
        self.reset_stream()
        return time_list, text

    # def inference(self, bgr, box, threshold=0.1):
    #     """
    #     input_tensor = image_processing.image_normalization(image,
    #                                                          mean=[0.485, 0.456, 0.406],
    #                                                          std=[0.229, 0.224, 0.225])
    #     input_tensor = input_tensor.transpose(2, 0, 1)  # [H0,W1,C2]-[C,H,W]
    #     input_tensor = torch.from_numpy(input_tensor)
    #     :param bgr:
    #     :param box:
    #     :param threshold:
    #     :return:
    #     """
    #     input_tensor, center, scale = self.pre_process(bgr, box)
    #     output = self.forward(input_tensor).cpu().numpy()
    #     kp_point, kp_score = self.post_process(output, center, scale, threshold)
    #     return kp_point, kp_score, self.skeleton
