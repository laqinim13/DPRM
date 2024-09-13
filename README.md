# Conformer-YOLO-HRNet: A Multimodal Framework for Dutar Performance Recognition

------

**Dutar**, as a traditional musical instrument, its learning method has historically relied mostly on oral transmission between masters and disciples. However, with the change of time, the teaching methods of Dutar have diversified, including traditional teacher-apprentice transmission, formal education in music schools, and the extensive use of online digital resources. Nonetheless, manual instruction is costly, and self-study lacks professional guidance. Despite the abundance of performance video resources on the Internet, the lack of complete tutorials or scores poses considerable difficulties for learners. In addition, most notation methods still rely on manual operation, which is not only time-consuming but also a waste of human resources. Therefore, this study proposes a **Conformer, HRNet, and YOLO-based Dutar performance recognition Multimodal model**, which aims to solve the above problems.

## The model is described and the code runs as follows（The video is in the “Model introduction video” folder）:
![](https://github.com/laqinim13/DPRM/blob/master/DPRM.gif)

# 配置环境

 - Anaconda 3
 - python3.8
 - Pytorch 1.9
 - Ubuntu 18.04

# 文件介绍
## conformer:语音识别模型代码

 -  conformer/configs：配置文件
 -  conformer/dataset:数据文件
 -  conformer/masr:模型文件
 -  conformer/model:权重文件
 -  conformer/tools，utils:辅助代码文件
 -  conformer/create_data.py:数据格式转换代码
 -  conformer/export_model:导出模型代码
 -  conformer/train.py:训练代码
 -  conformer/infer_path:测试代码

## yolov5:目标检测网络代码

 -  yolov5/data:目标检测对象配置和数据文件
 -  yolov5/models:yolo各个版本的配置文件
 -  yolov5/pretrained:预训练权重文件
 -  yolov5/runs:训练结果权重文件
 -  yolov5/utils:辅助代码文件
 -  yolov5/yolov5:模型文件
 -  yolov5/train.py:训练代码
 -  yolov5/val.py:验证代码
 -  yolov5/detec.py:测试代码

## 关键点检测

 - configs:配置文件
 - data:数据文件
 - work_space:训练结果保存文件
 - pose:模型代码
 - train.py:训练代码
 - test.py:测试代码
 
## 其他
 - my_test:测试视频文件
 - output:整体模型输出结果
 - demo.py:模型运行代码
 
# 数据标记
 - 目标检测和关键点检测数据标记
目标检测与关键点检测数据标注中分别对都塔尔、按压品位的手，扫弦的手，按压品位手的5个手指关键点及34个都塔尔品位关键点进行标注。（用labelme标注）
![]()

 - 语音识别标注
 声音识别数据标注中将向下扫弦和向上扫弦状态标注为“d”和“u”。
![]()

# 模型训练
## 目标检测网络yolov5的训练

 1. 先用【】使标注数据进行数据格式转换，从labelme生成的json文件中提取需要的数据，转换成yolov格式的txt数据
 2. 数据集格式(训练数据格式，测试跟这个一样)：
  /yolov5/data/train/
     - /images
        -- /1.jpg
        -- /2.jpg
        -- /....
     - /labels
        -- /1.txt
        -- /2.txt
        -- /...
     - /class.txt
 3. 设置完相应的配置后（配置文件已配置完成可以直接进行训练），用/yolov5/train.py进行训练
 4.训练结果：
![]()
## 关键点检测网络训练
### 手部关键点网络训练

 1. 先用【】使标注数据进行数据格式转换，从labelme生成的json文件中提取需要的数据，转换成yolov格式的txt数据
 2. 数据集格式：
  /yolov5/data
     - /images
        -- /1.jpg
        -- /2.jpg
        -- /....
     - /labels
        -- /1.txt
        -- /2.txt
        -- /...
     - /class.txt
 3. 设置完相应的配置后（配置文件已配置完成可以直接进行训练），用/yolov5/train.py进行训练
 4.训练结果：
![]()


 
