import argparse
import functools
import time
import wave

# from masr.predict import MASRPredictor
from conformer.masr.predict import MASRPredictor
from conformer.masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'conformer/configs/conformer.yml',     "配置文件")
add_arg('wav_path',         str,    'conformer/test_wav/02.wav',          "预测音频的路径")
add_arg('is_long_audio',    bool,   False,                       "是否为长语音")
add_arg('real_time_demo',   bool,   True,                       "是否使用实时语音识别演示")
add_arg('use_gpu',          bool,   False,                        "是否使用GPU预测")
add_arg('use_pun',          bool,   False,                       "是否给识别结果加标点符号")
add_arg('is_itn',           bool,   False,                       "是否对文本进行反标准化")
# add_arg('model_path',       str,    'models/conformer_streaming_fbank/inference.pt', "导出的预测模型文件路径")
add_arg('model_path',       str,    'conformer/models1/conformer_streaming_fbank/inference.pt', "导出的预测模型文件路径")
add_arg('pun_model_dir',    str,    'conformer/models/pun_models/',        "加标点符号的模型文件夹路径")
args = parser.parse_args()
print_arguments(args=args)
time_list =[]
len_1 = 0

# 获取识别器
predictor = MASRPredictor(configs=args.configs,
                          model_path=args.model_path,
                          use_gpu=args.use_gpu,
                          use_pun=args.use_pun,
                          pun_model_dir=args.pun_model_dir)


# 短语音识别
def predict_audio():
    start = time.time()
    result = predictor.predict(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)
    score, text = result['score'], result['text']
    print(f"消耗时间：{int(round((time.time() - start) * 1000))}ms, 识别结果: {text}, 得分: {int(score)}")


# 长语音识别
def predict_long_audio():
    start = time.time()
    result = predictor.predict_long(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)
    score, text = result['score'], result['text']
    print(f"长语音识别结果，消耗时间：{int(round((time.time() - start) * 1000))}, 得分: {score}, 识别结果: {text}")


# 实时识别模拟
def real_time_predict_demo():
    global time_list
    global len_1
    # 识别间隔时间
    interval_time = 0.05
    time_1 = 0
    time_list.append(time_1)
    CHUNK = int(16000 * interval_time)
    # 读取数据
    wf = wave.open(args.wav_path, 'rb')
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
        result = predictor.predict_stream(audio_data=data, use_pun=args.use_pun, is_itn=args.is_itn, is_end=d == b'',
                                          channels=channels, samp_width=samp_width, sample_rate=sample_rate)
        data = d
        if result is None:
            # time_list.append(time_1)
            time_1 = round(time_1 + 0.05, 2)
            # print(time_list)
            continue
        else:
            score, text = result['score'], result['text']
            text = text.replace("<unk>", "")
            a = len(text) - len_1
            len_1 = len(text)
            print(f"【实时结果】：消耗时间：{int((time.time() - start) * 1000)}ms, 识别结果: {text}, 得分: {int(score)}")
            time_1 = round(time_1 + 0.05, 2)
            if a > 0 :
                end_element = time_list[-1]
                qu_jian_zhi = time_1 - end_element
                junzhi_qujian = qu_jian_zhi / a
                for i in range(a):
                    time_list.append(round(end_element+junzhi_qujian*(i+1), 2))
            print(time_list)
    # 重置流式识别
    print()
    print(time_list)
    print(text)
    print(len(time_list))
    print(len(text))
    predictor.reset_stream()


if __name__ == "__main__":
    if args.real_time_demo:
        real_time_predict_demo()
    else:
        if args.is_long_audio:
            predict_long_audio()
        else:
            predict_audio()
