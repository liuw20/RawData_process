import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

from Network.CARS import CARS_Cloud
from Network.PLS import PLS_spectrum
from utils.loadData import RawData
from utils.loadData import Spetrum_dataloader


def SeetrumProcess():
    """Spectrum Process"""
    SeetrumPath = '../../Data/Grade2_down/Grape_Seetrum_3.24/GrapeSpectrum'  # spectrum数据地址
    labelPath = '../../Data/Grade2_down/Grape_Seetrum_3.24/label data.xlsx'  # label数据地址

    # spectrumPath = 'D:\FruitDetection\Data\Grade2_up\Test_data_total_Red'
    # labelPath = 'D:\FruitDetection\Data\Grade2_up\Test_data_total_Red\label_new_total.xlsx'

    # 读取要拟合的数据
    spectrum, label = Spetrum_dataloader(SeetrumPath, labelPath)

    # 对光谱数据进行特征波长的提取
    waveSelect, _ = CARS_Cloud(scale(spectrum), label)
    spectrum = spectrum[:, waveSelect]

    #  开始使用PLS进行建模
    # spectrum, label = kiwiData(spectrumPath, labelPath)
    RMSE, rpd = PLS_spectrum(spectrum, label)

    print("RMSE:{}  rpd:{}".format(RMSE, rpd))


def RawProcess():
    """RawData Process"""
    DataPath = ''
    RawImgPath = 'D:\FruitDetection\Data\Grade2_down\Grape_Seetrum_3.24\GrapeRaw'
    labelPath = '../../Data/Grade2_down/Grape_Seetrum_3.24/label data.xlsx'
    Tpath = 'D:\FruitDetection\Data\Grade2_down\I806S_T\coordinate_SOS-I806S_3x3_Spectrometer.txt'
    rawData, label = RawData(RawImgPath, labelPath, Tpath)  # 读取数据成功

    """拟合数据"""
    RMSE, rpd = PLS_spectrum(rawData, label,components=10)

    print("RMSE:{}   RPD:{}".format(RMSE, rpd))


def ComercialProcess():
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

    RMSE, rpd = PLS_spectrum(input_array, output_array,components=10)

    print(RMSE, rpd)



if __name__ == '__main__':
    # RawProcess()
    # SeetrumProcess()
    ComercialProcess()