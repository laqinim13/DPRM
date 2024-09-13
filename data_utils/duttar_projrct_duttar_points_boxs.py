import os
import json
import numpy as np
from pycocotools.coco import COCO

# with open('7.json', 'r', encoding='utf-8') as f:
#     labelme = json.load(f)
#
# labelme.keys()
# print(labelme['version'], labelme['flags'], labelme['imagePath'],
      # labelme['imageData'], labelme['imageHeight'], labelme['imageWidth'], labelme['shapes'])

coco = {}

coco['info'] = {}
coco['info']['description'] = 'ismayil&laqinim&duttar&No commercial use allowed'
coco['info']['year'] = 2024
coco['info']['date_created'] = '2024/02/01'

class_list2= {
    'supercategory': 'duttar',
    'id': 3,
    'name': 'duttar',
    'keypoints': ['duttar_1', 'duttar_2', 'duttar_3', 'duttar_4', 'duttar_5', 'duttar_6',
                  'duttar_7', 'duttar_8', 'duttar_9', 'duttar_10', 'duttar_11', 'duttar_12',
                  'duttar_13', 'duttar_14', 'duttar_15', 'duttar_16', 'duttar_17', 'duttar_18',
                  'duttar_19', 'duttar_20', 'duttar_21', 'duttar_22', 'duttar_23', 'duttar_24',
                  'duttar_25', 'duttar_26', 'duttar_27', 'duttar_28', 'duttar_29', 'duttar_30',
                  'duttar_31', 'duttar_32', 'duttar_33', 'duttar_34'],
    'skeleton': [[0,2], [2,4], [4,6], [6,8], [8,10], [10,12], [12,14], [14,16], [16,18], [18,20], [20,22], [22,24],
                 [24,26], [26,28], [28,30], [30,32],
                 [1,3], [3,5], [5,7], [7,9], [9,11], [11,13], [13,15], [15,17], [17,19], [19,21], [21,23], [23,25],
                 [25,27], [27,29], [29,31], [31,33]]
}

coco['categories'] = []
coco['categories'].append(class_list2)

coco['images'] = []
coco['annotations'] = []
IMG_ID = 1
ANN_ID = 1


def process_single_json(labelme, image_id=1):
    '''
    输入labelme的json数据，输出coco格式的每个框的关键点标注信息
    '''

    global ANN_ID

    coco_annotations = []

    for each_ann in labelme['shapes']:  # 遍历该json文件中的所有标注

        if each_ann['shape_type'] == 'rectangle':  # 筛选出框

            if each_ann['group_id'] == 3:

                # 该框元数据
                bbox_dict = {}
                bbox_dict['id'] = ANN_ID
                bbox_dict['image_id'] = image_id
                bbox_dict['category_id'] = 3
                bbox_dict['iscrowd'] = 0
                # print(ANN_ID)
                ANN_ID += 1
                # 获取该框坐标
                bbox_left_top_x = min(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
                bbox_left_top_y = min(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
                bbox_right_bottom_x = max(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
                bbox_right_bottom_y = max(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
                bbox_w = bbox_right_bottom_x - bbox_left_top_x
                bbox_h = bbox_right_bottom_y - bbox_left_top_y
                bbox_dict['bbox'] = [bbox_left_top_x, bbox_left_top_y, bbox_w, bbox_h]  # 左上角x、y、框的w、h
                bbox_dict['area'] = bbox_w * bbox_h

                # 筛选出该个体框中的所有关键点
                bbox_keypoints_dict = {}
                for each_ann2 in labelme['shapes']:  # 遍历所有标注
                    if each_ann2['shape_type'] == 'point':  # 筛选出关键点标注
                        if each_ann2['group_id'] == 3:
                            # 关键点横纵坐标
                            x = int(each_ann2['points'][0][0])
                            y = int(each_ann2['points'][0][1])
                            label = each_ann2['label']
                            if (x > bbox_left_top_x) & (x < bbox_right_bottom_x) & (y < bbox_right_bottom_y) & (
                                    y > bbox_left_top_y):  # 筛选出在该个体框中的关键点
                                bbox_keypoints_dict[label] = [x, y]

                bbox_dict['num_keypoints'] = len(bbox_keypoints_dict)
                # print(bbox_keypoints_dict)

                # 把关键点按照类别顺序排好
                bbox_dict['keypoints'] = []
                for each_class in class_list2['keypoints']:
                    if each_class in bbox_keypoints_dict:
                        bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0])
                        bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1])
                        bbox_dict['keypoints'].append(1)  # 2-可见不遮挡 1-遮挡 0-没有点
                    else:  # 不存在的点，一律为0
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)

                coco_annotations.append(bbox_dict)
    return coco_annotations
# print(process_single_json(labelme))

# images和annotations
# 遍历所有 labelme 格式的 json 文件
def mian_1():
    global IMG_ID
    folder_path = 'E:/pyCharmProjec/shipin_chai_fen/labelme2coco/test_json/test'
    for labelme_json in os.listdir(folder_path):
        if labelme_json.split('.')[-1] == 'json':
            labelme_json = os.path.join(folder_path,labelme_json)
            with open(labelme_json, 'r', encoding='utf-8') as f:
                labelme = json.load(f)

                ## 提取图像元数据
                img_dict = {}
                img_dict['license'] = 1
                img_dict['file_name'] = labelme['imagePath']
                img_dict['coco_url'] = ""
                img_dict['height'] = labelme['imageHeight']
                img_dict['width'] = labelme['imageWidth']
                img_dict['data_captured'] = "2024/02/01"
                img_dict['flickr_url'] = ""
                img_dict['id'] = IMG_ID
                coco['images'].append(img_dict)

                ## 提取框和关键点信息
                coco_annotations = process_single_json(labelme, image_id=IMG_ID)
                coco['annotations'] += coco_annotations

                IMG_ID += 1

                print(labelme_json, '已处理完毕')

        else:
            pass
    # 4 全部 手
    # 5全部 都塔尔
    if not os.path.exists('E:/pyCharmProjec/shipin_chai_fen/labelme2coco/test_json/output_coco5'):
        os.mkdir('E:/pyCharmProjec/shipin_chai_fen/labelme2coco/test_json/output_coco5')
        print('创建新目录 output_coco2')

    coco_path = 'E:/pyCharmProjec/shipin_chai_fen/labelme2coco/test_json/output_coco5/test_anno.json'
    with open(coco_path, 'w') as f:
        json.dump(coco, f, indent=2)

if __name__ == '__main__':
    mian_1()


# my_coco = COCO(coco_path)