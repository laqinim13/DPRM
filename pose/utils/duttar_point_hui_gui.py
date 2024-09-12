import torch
import numpy as np
import matplotlib.pyplot as plt


def duttar_point_hui_gui(data):
    # # 计算平均值
    # # print(data[::2, 0])
    # data = np.array(data)
    # data = np.squeeze(data)
    # data = np.squeeze(data)
    if data[0].size > 0:
        x1 = data[0][::2, 0]
        mean_x = np.mean(data[0][::2, 0])
        # print(mean_x)
        mean_y = np.mean(data[0][::2, 1])
        # print(mean_y)

        # 计算误差
        error_x = data[0][::2, 0] - mean_x
        error_y = data[0][::2, 1] - mean_y

        # 计算平方误差
        squared_error_x = np.square(error_x)
        # squared_error_y = np.square(error_y)

        # 计算误差乘积和
        error_product_sum = np.sum(error_x * error_y)
        # 计算 x 的平方误差的和
        squared_error_sum_x = np.sum(squared_error_x)
        b = error_product_sum / squared_error_sum_x
        b1 = mean_y - b * mean_x
        y = b * x1 + b1
        # print(f"y = x * {b} + {b1}")
        # print("y")
        # print(y)
        # print()
        data[0][::2, 1] = y
        # print(data[0])

        x2 = data[0][1::2, 0]
        # 计算平均值
        # print(data[0][1::2, 0])
        mean_x1 = np.mean(data[0][1::2, 0])
        # print(mean_x1)
        mean_y1 = np.mean(data[0][1::2, 1])
        # print(mean_y1)

        # 计算误差
        error_x1 = data[0][1::2, 0] - mean_x1
        error_y1 = data[0][1::2, 1] - mean_y1

        # 计算平方误差
        squared_error_x1 = np.square(error_x1)
        # squared_error_y1 = np.square(error_y1)

        # 计算误差乘积和
        error_product_sum1 = np.sum(error_x1 * error_y1)
        # 计算 x 的平方误差的和
        squared_error_sum_x1 = np.sum(squared_error_x1)
        b2 = error_product_sum1 / squared_error_sum_x1
        b21 = mean_y1 - b2 * mean_x1
        y2 = b2 * x2 + b21
        data[0][1::2, 1] = y2
        # print(f"y = x * {b2} + {b21}")
    # print(data[0])
    return data, b, b1



