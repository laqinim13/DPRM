import matplotlib.pyplot as plt
import numpy as np


def pada(hand, duttar, m, b):
    # 提取单数行的第一个元素
    x = duttar[0][::2, 0]
    if len(duttar) >= 1:
        points_numpy = np.array(duttar)
        points = points_numpy.reshape(-1, 2, 2)
        distances = np.linalg.norm(points[:, 1, :] - points[:, 0, :], axis=1)
        average_distance = np.mean(distances)
        distances = np.insert(distances, 0, average_distance)

        single_rows = points_numpy[0][::2]
        distances1 = np.linalg.norm(single_rows[1:] - single_rows[:-1], axis=1)
        average_distance1 = np.mean(distances1)
        distances1 = np.insert(distances1, 0, average_distance1)
    # print(x[0])

    # pada_list = ['高5', '升高4', '高4', '高3', '升高2', '高2', '升高1', '高1',
    #              '7', '升6', '6', '升5', '5', '升4', '4', '3' '升2']
    pada_list = ['#2', '3', '4', '#4', '5', '#5', '6', '#6',
                 '7', 'up1', '#up1', 'up2', '#up2', 'up3', 'up4', '#up4' 'up5']

    barmak_list = ['大拇指', '食指', '中指', '无名指', '小值']
    # 定义直线函数
    def line(x, m, b):
        return m * x + b
    # 直线参数
    a = max(x) + 50
    # 自定义分割点的x坐标
    x_divisions = x
    # x_divisions = np.append(x_divisions, a)
    x_divisions = np.insert(x_divisions, 0, a)

    # 判断点是否在圆内
    # barmak = -1
    pada = []
    one_pada = []
    for j in range(len(hand)):
    # for barmak, row in enumerate(hand[j]):
        for barmak, row in enumerate(hand[j]):
            p_x = row[0]
            p_y = row[1]
            i = 0
            for i, x1 in enumerate(x_divisions):
                y = line(p_x, m, b)
                if i < 17:
                    if np.logical_and(x_divisions[i + 1] - (distances1[i]/4) < p_x, p_x <= x_divisions[i] - (distances1[i]/4)):
                        if y - (distances[i]/3) <= p_y <= y + (distances[i]/3):
                            one_pada.append(i)
                            # print(f"{barmak_list[barmak]}在第【 {pada_list[i]} 】个区域内内。")

    if len(one_pada) != 0:
        pada.append(pada_list[max(one_pada)])
    return pada
