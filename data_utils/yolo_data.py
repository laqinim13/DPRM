import os
import json
import shutil
import numpy as np
from tqdm import tqdm

# Dataset_root = 'Ruler_15_Dataset'

classes = {
    'up_hand':0,
    'down_hand':1,
    'duttar':2,
}

print(list(classes.keys()))

# os.chdir(Dataset_root)
with open('E:/pyCharmProjec/shipin_chai_fen/labelme2coco/duttar_projrct_all_boxs/one/classes.txt', 'w', encoding='utf-8') as f:
    for each in list(classes.keys()):
        f.write(each + '\n')


# os.mkdir('labels')
# os.mkdir('labels/train')
# os.mkdir('labels/val')


def process_single_json(labelme_path, save_folder):
    # 载入 labelme格式的 json 标注文件
    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    img_width = labelme['imageWidth']  # 图像宽度
    img_height = labelme['imageHeight']  # 图像高度

    # 生成 YOLO 格式的 txt 文件
    suffix = labelme_path.split('.')[-2]
    yolo_txt_path = suffix + '.txt'

    with open(yolo_txt_path, 'w', encoding='utf-8') as f:
        for each_ann in labelme['shapes']:  # 遍历每个框

            if each_ann['shape_type'] == 'rectangle':  # 筛选出框

                # 获取类别 ID
                bbox_class_id = classes[each_ann['label']]

                # 左上角和右下角的 XY 像素坐标
                bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))

                # 框中心点的 XY 像素坐标
                bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)

                # 框宽度
                bbox_width = bbox_bottom_right_x - bbox_top_left_x

                # 框高度
                bbox_height = bbox_bottom_right_y - bbox_top_left_y

                # 框中心点归一化坐标
                bbox_center_x_norm = bbox_center_x / img_width
                bbox_center_y_norm = bbox_center_y / img_height

                # 框归一化宽度
                bbox_width_norm = bbox_width / img_width
                # 框归一化高度
                bbox_height_norm = bbox_height / img_height

                # 生成 YOLO 格式的一行标注，指定保留小数点后几位
                bbox_yolo_str = '{} {:.4f} {:.4f} {:.4f} {:.4f}'.format(bbox_class_id, bbox_center_x_norm,
                                                                        bbox_center_y_norm, bbox_width_norm,
                                                                        bbox_height_norm)
                # 写入 txt 文件中
                f.write(bbox_yolo_str + '\n')

    shutil.move(yolo_txt_path, save_folder)
    print('{} --> {} 转换完成'.format(labelme_path, yolo_txt_path))

if __name__ == '__main__':
    #   转换训练集标注文件至labels/train目录
    os.chdir('E:/pyCharmProjec/shipin_chai_fen/1/4_1/val/img')
    save_folder = 'E:/pyCharmProjec/shipin_chai_fen/1/4_1/val'
    for labelme_path in os.listdir():
        process_single_json(labelme_path, save_folder=save_folder)
    print('YOLO格式的txt标注文件已保存至 ', save_folder)

# # 转换测试集标注文件至labels/val目录
# os.chdir('labelme_jsons/val')
# save_folder = '../../labels/val'
# for labelme_path in os.listdir():
#     process_single_json(labelme_path, save_folder=save_folder)
# print('YOLO格式的txt标注文件已保存至 ', save_folder)

