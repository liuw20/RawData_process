import os

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt


def Spetrum_dataloader(SeetrumPath, labelPath):
    files = os.listdir(SeetrumPath)
    spetrumList = []
    for file in files:
        # temp=[]
        if not os.path.isdir(file):
            f = pd.read_csv(SeetrumPath + '/' + file, skiprows=16)
            temp = f.to_numpy()
            waveLength = temp[:, 0]
            temp = temp[:, 1::2]  # 取数组的奇数列
            temp = np.mean(temp, axis=1)  # 按行求平均
            spetrumList.append(temp)  # 得到了所有文件的数据

    spetrumArray = np.array(spetrumList)
    label = pd.read_excel(labelPath, header=None)
    label = np.array(label)  # 将DataFrame转化为numpy数据
    label = label[:, 1]
    label = label[:, np.newaxis]

    return spetrumArray, label  # 返回数据


def kiwiData(spectrumPath, labelPath):
    f = pd.read_excel(spectrumPath + '\\' + 'Total data.xlsx')
    temp = f.to_numpy()
    spectrum = temp[:, 4:]  # 只取部分光谱数据
    spectrum = spectrum.T

    label = pd.read_excel(labelPath, header=None)
    label = np.array(label)
    label = label[1, :].T
    label = label[:, np.newaxis]

    return spectrum, label


def RawData(RawImgPath, labelPath, Tpath):
    T = np.loadtxt(Tpath)
    T = T.astype(int)
    Row = T[:, 0]
    Col = T[:, 1]
    files = os.listdir(RawImgPath)
    files = list(filter(lambda x: x.endswith('.bmp'), files))  # 实际就是Raw图的格式，但是缺乏帧头的信息
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(RawImgPath, x)))
    RawData = []
    for file in files:
        if not os.path.isdir(file):
            Img = Image.open(RawImgPath + '\\' + file)
            image = np.array(Img)
            temp = []
            for i in range(len(T)):
                temp.append(image[T[i][0], T[i][1]])
            # temp = image[Row, :]
            # temp = temp[:, Col]
            # temp = temp.flatten()
            RawData.append(temp)  # 此时将该数据拉平后并进行了去重或者不去去重
    RawData = np.array(RawData)  # 得到了去重后的数据

    """读取标签"""
    label = pd.read_excel(labelPath, header=None)
    label = np.array(label)  # 将DataFrame转化为numpy数据
    label = label[:, 1]
    label = label[:, np.newaxis]

    return RawData, label


def Tmatrix(Tpath):
    T = np.load(Tpath + '\T_I806S_3x3_avg.npz')
    T_data = T['T']
    waveLength = np.arange(380, 851)
    f = plt.figure()
    # fig, ax = plt.subplots(5,6)

    for i in range(120,150):
        y = T_data[:, i]
        f.add_subplot(5, 6, i+1-120)
        plt.plot(waveLength, y)
        plt.xticks([])
        plt.yticks([])

        # row = i // 6
        # col = i % 6

        # plt.plot(waveLength,y)
        # ax[row][col].plot(waveLength, y)
        # ax[row][col].xticks([])

    # plt.xlabel('Wavelength')
    # plt.ylabel('Intensity')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.title('不同结构的透射谱')
    plt.show()


if __name__ == '__main__':
    Tpath = 'D:\FruitDetection\Data\Grade2_down\I806S_T'
    Tmatrix(Tpath)
    # labelPath = '../../../Data/Grade2_down/Grape_Seetrum_3.24/label data.xlsx'
# label=label_dataloder(labelPath)

# spectrumPath = 'D:\FruitDetection\Data\Grade2_up\Test_data_total_Red'
# labelPath = 'D:\FruitDetection\Data\Grade2_up\Test_data_total_Red\label_new_total.xlsx'
# spectrum, label = kiwiData(spectrumPath, labelPath)
