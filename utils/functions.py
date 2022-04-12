import sys

import numpy as np
import pandas as pd

"""定义一些常用的函数"""


def RPD(label, RMSEP):
    std = np.std(label)
    RPD = std / RMSEP
    return RPD


def ks_partition(specs, test_rate=0):
    '''
    实现KS法用来划分数据集
    :param specs: 光谱数据mxn
    :param test_rate: 测试集比例
    :return: 返回的是训练集的list
    '''
    M = specs.shape[0]
    N = round(M * (1 - test_rate))
    samples = np.arange(M)

    distance = np.zeros((M, M))

    # 计算所有光谱两两之间的距离
    for i in range(M):
        x_a = specs[i]
        for j in range(0, M):
            x_b = specs[j]
            distance[i, j] = np.linalg.norm(x_a - x_b)

    delete_indexs = []
    max_distance = np.array([d.max() for d in distance])  # 每一个光谱到其余光谱之间距离最大的值
    max_first = np.where(max_distance == max(max_distance))[0][0]  # 最大距离里面的第一大
    delete_indexs.append(max_first)
    result_samples = np.delete(samples, delete_indexs)
    max_distance[max_first] = 0
    max_second = np.where(max_distance == max(max_distance))[0][0]  # 最大距离里面的第二大
    delete_indexs.append(max_second)
    result_samples = np.delete(samples, delete_indexs)

    m = np.zeros(N).astype(int)
    m[0] = max_first
    m[1] = max_second

    # 依次选择剩下的N-2个样本
    for i in range(2, N):
        # 对于每个需要选择的样本来说，应该是遍历剩下的样本到已选择样本的最短距离，选其中最大的
        # 这里用result_sample记录剩下的样本，用delete_indexs记录选择的样本
        min_d_list = np.zeros(M)
        for j in range(M - i):
            min_d = sys.maxsize
            for k in range(i):
                min_d = min(min_d, distance[result_samples[j], m[k]])
            min_d_list[result_samples[j]] = min_d
        max_i = np.where(min_d_list == max(min_d_list))[0][0]
        # if i == N - 1:
        #     print(result_samples)
        #     print(min_d_list)
        delete_indexs.append(max_i)
        result_samples = np.delete(samples, delete_indexs)
        m[i] = max_i

    return delete_indexs



if __name__ == '__main__':
    input = pd.read_excel("D:\FruitDetection\Data\Grade2_up\Test_data_total_Red\Total data.xlsx")
    input_array = input.to_numpy()
    input_array = np.delete(input_array, [0, 1, 2, 3], 1)
    input_array = input_array.T
    delete_index= ks_partition(input_array)
