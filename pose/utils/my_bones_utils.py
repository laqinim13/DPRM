# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2021-11-03 14:38:51
    @Brief  : 颜色表： https://www.rapidtables.org/zh-CN/web/color/RGB_Color.html
"""
# skeleton连接线，keypoint关键点名称，num_joints关键点个数
BONES = {
    "coco_person": {
        # "skeleton": [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
        #              [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]],
        "skeleton": [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                     [6, 8], [7, 9], [8, 10], [0, 1], [0, 2], [1, 3], [2, 4]],
        "keypoint": [],
        "num_joints": 17,
        "names": {0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear", 5: "left_shoulder",
                  6: "right_shoulder", 7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 10: "right_wrist",
                  11: "left_hip", 12: "right_hip", 13: "left_knee", 14: "right_knee", 15: "left_ankle",
                  16: "right_ankle"},
        "colors": [
            [0, 51, 102], [0, 0, 102], [51, 0, 102], [0, 0, 153], [76, 0, 153],
            [0, 0, 204], [102, 0, 204], [0, 0, 255], [127, 0, 255], [51, 51, 255], [153, 51, 255],
            [0, 102, 204], [204, 0, 204], [0, 128, 255], [255, 0, 255], [51, 153, 255], [255, 51, 255], [153, 0, 0]
        ]
    },
    "mpii": {
        "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                     [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
                     [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]],
        "keypoint": [],
        "num_joints": 21,
        "names": {0: "rankle", 1: "r knee", 2: "r hip", 3: "l hip", 4: "l knee", 5: "l ankle", 6: "pelvis",
                  7: "thorax", 8: "upper neck", 9: "head top", 10: "r wrist", 11: "r wrist", 12: "r shoulder",
                  13: "l shoulder", 14: "l elbow", 15: "l wrist"},
        "colors": None
    },
    "hand": {
        "skeleton": [],
        "keypoint": [],
        "num_joints": 5,
        "names": {},
        "colors": None

    },
    "duttar": {
        "skeleton": [[0, 2],[2, 4],[4, 6],[6, 8],[8,10],[10,12],[12,14],[14, 16],[16,18],[18,20],[20,22],
                     [22,24],[24,26],[26,28],[28,30],[30,32],[1,3],[3,5],[5,7],[7,9],[9,11],[11,13],[13,15],
                     [15,17],[17,19],[19,21],[21,23],[23,25],[25,27],[27,29],[29,31],[31,33]],
        "keypoint": [],
        "num_joints": 34,
        "names": {},
        "colors": None

    },
}


def get_target_bones(target):
    return BONES.get(target, {})
