import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from utils.functions import RPD


def PC_Cross_Validation(X, y, pc, cv):
    '''
        x :光谱矩阵 nxm
        y :浓度阵 （化学值）
        pc:最大主成分数
        cv:交叉验证数量
    return :
        RMSECV:各主成分数对应的RMSECV
        PRESS :各主成分数对应的PRESS
        rindex:最佳主成分数
    '''
    kf = KFold(n_splits=cv)  # 选定交叉验证方式
    RMSECV = []
    for i in range(pc):
        RMSE = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pls = PLSRegression(n_components=i + 1)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        RMSE_mean = np.mean(RMSE)
        RMSECV.append(RMSE_mean)
    rindex = np.argmin(RMSECV)
    return RMSECV, rindex


def Cross_Validation(X, y, pc, cv):
    '''
     x :光谱矩阵 nxm
     y :浓度阵 （化学值）
     pc:最大主成分数
     cv:交叉验证数量
     return :
            RMSECV:各主成分数对应的RMSECV
    '''
    kf = KFold(n_splits=cv)
    RMSE = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pls = PLSRegression(n_components=pc)
        pls.fit(x_train, y_train)
        y_predict = pls.predict(x_test)
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
    RMSE_mean = np.mean(RMSE)
    return RMSE_mean


def CARS_Cloud(X, y, N=50, f=10, cv=5):
    '''
    CARS算法流程
    :param X: 光谱矩阵
    :param y: 理化值数据
    :param N: 采样次数
    :param f: 主成分数量
    :param cv: cv折交叉验证
    :return: 最佳波长的索引
    '''

    p = 0.8  # 选择80%数据集进入训练集
    m, n = X.shape
    u = np.power((n / 2), (1 / (N - 1)))
    k = (1 / (N - 1)) * np.log(n / 2)
    cal_num = np.round(m * p)  # 校正集数量
    # val_num = m - cal_num
    b2 = np.arange(n)  # 创建等差数列
    x = copy.deepcopy(X)  # 拷贝副本
    D = np.vstack((np.array(b2).reshape(1, -1), X))  # 垂直堆叠数据
    WaveData = []
    # Coeff = []
    WaveNum = []
    RMSECV = []
    r = []
    for i in range(1, N + 1):
        r.append(u * np.exp(-1 * k * i))  # EDF筛选波长比例
        wave_num = int(np.round(r[i - 1] * n))  # 剩余波长数量
        WaveNum = np.hstack((WaveNum, wave_num))  # 水平方向上平铺
        cal_index = np.random.choice(np.arange(m), size=int(cal_num), replace=False)  # 对校正集打乱了顺序

        wave_index = b2[:wave_num].reshape(1, -1)[0]  # 波长的索引
        xcal = x[np.ix_(list(cal_index), list(wave_index))]  # 取得波长和样本的对应矩阵
        # xcal = xcal[:,wave_index].reshape(-1,wave_num)
        ycal = y[cal_index]  #
        x = x[:, wave_index]
        D = D[:, wave_index]
        d = D[0, :].reshape(1, -1)
        wnum = n - wave_num
        if wnum > 0:
            d = np.hstack((d, np.full((1, wnum), -1)))
        if len(WaveData) == 0:
            WaveData = d
        else:
            WaveData = np.vstack((WaveData, d.reshape(1, -1)))

        if wave_num < f:
            f = wave_num

        pls = PLSRegression(n_components=f)
        pls.fit(xcal, ycal)
        beta = pls.coef_  # 每个波长的回归系数
        b = np.abs(beta)
        b2 = np.argsort(-b, axis=0)  # 提取索引  由大到小排列
        coef = copy.deepcopy(beta)
        coeff = coef[b2, :].reshape(len(b2), -1)
        # cb = coeff[:wave_num]
        #
        # if wnum > 0:
        #     cb = np.vstack((cb, np.full((wnum, 1), -1)))
        # if len(Coeff) == 0:
        #     Coeff = copy.deepcopy(cb)
        # else:
        #     Coeff = np.hstack((Coeff, cb))
        rmsecv, rindex = PC_Cross_Validation(xcal, ycal, f, cv)
        RMSECV.append(Cross_Validation(xcal, ycal, rindex + 1, cv))
    # CoeffData = Coeff.T

    WAVE = []
    # COEFF = []

    for i in range(WaveData.shape[0]):
        wd = WaveData[i, :]
        # cd = CoeffData[i, :]
        WD = np.ones((len(wd)))
        # CO = np.ones((len(wd)))
        for j in range(len(wd)):
            ind = np.where(wd == j)
            if len(ind[0]) == 0:
                WD[j] = 0
                # CO[j] = 0
            else:
                WD[j] = wd[ind[0]]
                # CO[j] = cd[ind[0]]
        if len(WAVE) == 0:
            WAVE = copy.deepcopy(WD)
        else:
            WAVE = np.vstack((WAVE, WD.reshape(1, -1)))
        # if len(COEFF) == 0:
        #     COEFF = copy.deepcopy(CO)
        # else:
        #     COEFF = np.vstack((WAVE, CO.reshape(1, -1)))

    MinIndex = np.argmin(RMSECV)
    minRMSECV = np.min(RMSECV)
    Optimal = WAVE[MinIndex, :]
    boindex = np.where(Optimal != 0)
    OptWave = boindex[0]

    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fonts = 16
    plt.subplot(211)
    # plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    # plt.ylabel('被选择的波长数量', fontsize=fonts)
    plt.title('最佳迭代次数：' + str(MinIndex) + '次', fontsize=fonts)
    plt.plot(np.arange(N), WaveNum)

    plt.subplot(212)
    plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    plt.ylabel('RMSECV', fontsize=fonts)
    plt.plot(np.arange(N), RMSECV)

    # # plt.subplot(313)
    # # plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    # # plt.ylabel('各变量系数值', fontsize=fonts)
    # # plt.plot(COEFF)
    # #plt.vlines(MinIndex, -1e3, 1e3, colors='r')
    plt.show()

    return OptWave, minRMSECV


if __name__ == '__main__':

    '''
    在规模数据集上其实是可以的，但是在小规模数据上并没有什么优势
    '''
    input = pd.read_excel("D:\FruitDetection\Data\Grade2_up\Test_data_total_Red\Total data.xlsx")
    input_array = input.to_numpy()
    input_array = np.delete(input_array, [0, 1, 2, 3], 1)
    input_array = input_array.T

    output = pd.read_excel("D:\FruitDetection\Data\Grade2_up\Test_data_total_Red\label_new_total.xlsx")
    output_array = output.to_numpy()
    output_array = np.delete(output_array, [1, 2], 0)
    # output_array = output_array[~np.isnan(output_array)]
    output_array = output_array.T

    Optwave, minRMSECV = CARS_Cloud(scale(input_array), output_array)
    input_array = input_array[:, Optwave]
    X_train, X_test, y_train, y_test = train_test_split(input_array, output_array, test_size=0.2, random_state=0)
    pls = PLSRegression(n_components=10)  # 这个根据曲线来判断
    pls.fit(scale(X_train), y_train)
    RMSE = np.sqrt(mean_squared_error(y_test, pls.predict(scale(X_test))))
    rpd=RPD(output_array,RMSE)

    print(RMSE,rpd)
