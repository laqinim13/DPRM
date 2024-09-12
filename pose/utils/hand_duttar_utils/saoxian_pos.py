import argparse
import functools
import time
import wave
from pose.utils.hand_duttar_utils.saoxian_demo import MASRPredictor

time_list = []
len_1 = 0
class Saoxian_Pose():
    def __init__(self):
        self.is_long_audio = False
        self.real_time_demo = True
        configs = "conformer/configs/conformer.yml"
        wav_path = "my_test/test_wav.wav"
        use_gpu = True
        is_itn = False
        use_pun = False
        model_path = "conformer/models1/conformer_streaming_fbank/inference.pt"
        pun_model_dir = "conformer/models/pun_models/"
        self.detector = MASRPredictor(configs = configs,
                                       model_path=model_path,
                                       use_gpu=use_gpu,
                                       pun_model_dir=pun_model_dir,
                                       use_pun=use_pun,
                                       wav_path=wav_path,
                                       is_itn=is_itn
                                       )

    def detect(self, real_time_demo=True, use_pun=False, is_itn=False, is_long_audio=False, wav_path="my_test/test_wav.wav"):
        """

        :param real_time_demo:
        :param use_pun:
        :param is_itn:
        :param is_long_audio:
        :param wav_path:
        :return:
        """
        if real_time_demo:
            time_list, text = self.detector.real_time_predict_demo(wav_path="my_test/test_wav.wav", use_pun=False, is_itn=False)
        else:
            if is_long_audio:
                time_list, text =self.detector.predict_long_audio(wav_path, use_pun, is_itn)
            else:
                time_list, text =self.detector.predict_audio(wav_path, use_pun, is_itn)
        return time_list, text


