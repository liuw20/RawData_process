import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from utils.functions import RPD

def PLS_spectrum(spectrum, label,components):
    """拟合数据"""
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)  # 设置交叉验证
    mse = []
    n = len(spectrum)

    # 计算score 挑选合适的主成分
    # score = -1 * model_selection.cross_val_score(PLSRegression(n_components=1),
    #                                              scale(np.ones((n, 1))) , label, cv=cv,
    #                                              scoring='neg_mean_squared_error').mean()  # 只使用截距计算
    # mse.append(score)
    '''寻找最佳的主成分数量'''
    # for i in np.arange(1, 20):
    #     pls = PLSRegression(n_components=i)  # 设置主成分数量
    #     score = -1 * model_selection.cross_val_score(pls, scale(spectrum), label, cv=cv,
    #                                                  scoring='neg_mean_squared_error').mean()
    #     mse.append(score)
    # plt.plot(mse)
    # plt.xlabel("Number of PLS Components")
    # plt.ylabel('MSE')
    # plt.title('Grape SSC')
    # plt.show()

    # 训练集和测试集的划分
    X_train, X_test, y_train, y_test = train_test_split(spectrum, label, test_size=0.3, random_state=0)

    # 给定样本集均值的预测指标效果
    mean=np.mean(y_train)
    mean_pre=np.ones_like(y_test)
    mean_pre=mean_pre*mean
    RMSE_mean=(mean_squared_error(y_test, mean_pre))
    rpd_mean=RPD(label,RMSE_mean)
    print("RMSE_mean:{},mean_RPD:{}\n".format(RMSE_mean,rpd_mean))

    # 计算RMSE
    pls = PLSRegression(n_components=components)  # 这个根据曲线来判断
    pls.fit(scale(X_train), y_train)
    RMSEP= np.sqrt(mean_squared_error(y_test, pls.predict(scale(X_test))))
    rpd=RPD(label,RMSEP)

    return RMSEP,rpd
