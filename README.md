# Conformer-YOLO-HRNet: A Multimodal Framework for Dutar Performance Recognition

------

**Dutar**, as a traditional musical instrument, its learning method has historically relied mostly on oral transmission between masters and disciples. However, with the change of time, the teaching methods of Dutar have diversified, including traditional teacher-apprentice transmission, formal education in music schools, and the extensive use of online digital resources. Nonetheless, manual instruction is costly, and self-study lacks professional guidance. Despite the abundance of performance video resources on the Internet, the lack of complete tutorials or scores poses considerable difficulties for learners. In addition, most notation methods still rely on manual operation, which is not only time-consuming but also a waste of human resources. Therefore, this study proposes a **Conformer, HRNet, and YOLO-based Dutar performance recognition Multimodal model**, which aims to solve the above problems.

## The model is described and the code runs as follows（The video is in the “Model introduction video” folder）:
![](https://github.com/laqinim13/DPRM/blob/master/DPRM.gif)

# Configure the environment

 - Anaconda 3
 - Python 3.8
 - Pytorch 1.9
 - Ubuntu 18.04

# Documentation
## conformer: speech recognition model code

 - conformer/configs: configuration files
 - conformer/dataset:data files
 - conformer/masr:model files
 - conformer/model:weights file
 - conformer/tools, utils: auxiliary code files
 - conformer/create_data.py: data format conversion code.
 - conformer/export_model:Export model code.
 - conformer/train.py: training code.
 - conformer/infer_path:test code.

## yolov5:Target detection network code.

 - yolov5/data:Target detection object configuration and data files.
 - yolov5/models:Configuration files for each version of yolo.
 - yolov5/pretrained:pre-training weights file
 - yolov5/runs: training result weights file.
 - yolov5/utils: auxiliary code files.
 - yolov5/yolov5:model file
 - yolov5/train.py: training code
 - yolov5/val.py:validation code
 - yolov5/detec.py:test code

## Keypoint detection

 - configs: configuration file
 - data:data file
 - work_space: save the training result file.
 - pose:model code
 - train.py: training code.
 - test.py: test code.
 
## Others
 - my_test:test video file.
 - output:overall model output
 - demo.py: model running code.
 
# Data labeling
## Target detection and keypoint detection data labeling
The target detection and keypoint detection data labeling are labeled for the 5 finger keypoints of the dutar, the hand that presses the taste, the hand that sweeps the strings, the hand that presses the taste, and the 34 keypoints of the dutar's taste, respectively. (Labeled with labelme)
![](https://github.com/laqinim13/DPRM/blob/master/images/label.png)

## Voice Recognition Labeling
 Voice recognition data labeled with “d” and “u” for downward and upward sweeps.
![](https://github.com/laqinim13/DPRM/blob/master/images/qupu.png)

# Model training
## Overall model results
![](https://github.com/laqinim13/DPRM/blob/master/images/jieguo.png)
## Training of target detection network yolov5

 1. first use [ ] to make the labeled data for data format conversion, from the labelme generated json file to extract the required data, converted to yolov format txt data
 2. dataset format (training data format, test with this same):
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
 3. After setting up the corresponding configuration (configuration file has been configured can be directly for training), with /yolov5/train.py for training
 4. Training results:
![](https://github.com/laqinim13/DPRM/blob/master/images/yolo.png)

## Keypoint detection network training ##
### Hand keypoint network training

 1. first use [ ] to make the labeled data for data format conversion, from the labelme generated json file to extract the required data, converted to the corresponding hand key point data format txt data
 2. Data set format:
  /data
     - /train
        -- /images
        -- /train.json
        -- /....
     - /test
        -- /images
        -- /test.json
        -- /...
     - /class.txt
 3. After setting up the corresponding hand keypoint configuration (configuration file has been configured can be trained directly), use train.py to train (training with the hand keypoint configuration file)
 4. Training results:
![](https://github.com/laqinim13/DPRM/blob/master/images/hand.png)
### Dutar keypoint network training

 1. first use [ ] to make the labeled data for data format conversion, from the labelme generated json file to extract the required data, converted into corresponding to the key point data format of the Duetal txt data
 2. dataset format:
  /data
     - /train
        -- /images
        -- /train.json
        -- /....
     - /test
        -- /images
        -- /test.json
        -- /...
     - /class.txt
 3. After setting up the corresponding Dutal key point configuration (configuration file has been configured can be directly for training), use train.py for training (training with Dutal key point configuration file)
 4. Training results:
![](https://github.com/laqinim13/DPRM/blob/master/images/dutar.png)

## Sweep string state recognition module training
 1. first use [ ] to make the labeled data for data format conversion, from the json file generated by labelme to extract the required data, converted into txt data corresponding to the Dutal key point data format
 2. dataset format:
  /data
     - /train
        -- /images
        -- /train.json
        -- /....
     - /test
        -- /images
        -- /test.json
        -- /...
     - /class.txt
 3. After setting up the corresponding Dutal key point configuration (configuration file has been configured can be directly for training), use train.py for training (training with Dutal key point configuration file)
 4. Training results:
![]()
