import yaml
import logging
import numpy as np
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from dataloader import UXODataset


feature_list = [
    'I_425Hz', 
    'Q_425Hz', 
    'I_1525Hz', 
    'Q_1525Hz', 
    'I_5325Hz', 
    'Q_5325Hz', 
    'I_18325Hz', 
    'Q_18325Hz', 
    'I_63025Hz', 
    'Q_63025Hz'
    ]

classes2fileNum = {
    0: ['096', '099', '098', '097', '102', '103', '104', '119'],
    1: ['085', '086', '080', '081', '082', '083', '084'],
    2: ['069', '071', '070', '072', '073', '074', '076', '077', '078', '075', '107', '108', '111', '112', '113', '114', '115', '120', '122', '123', '124', '125', '127', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140']
}

fileNum2classes = {}
for classes, fileNums in classes2fileNum.items():
    for f in fileNums:
        fileNum2classes[f] = classes

def seconds2hours(s):
    minutes = int(s / 60)
    seconds = int(s - minutes * 60)
    hours = int(minutes / 60)
    minutes = minutes - hours * 60
    return hours, minutes, seconds


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def log_config(config, logger):
    for section in config:
        logger.info(f"Section: {section}")
        for key, value in config[section].items():
            logger.info(f"{key}: {value}")


def normalize(data):
    # data: Batch size * Window size * Feature num
    config = load_config('config.yaml')
    min_val = config['training']['min_val']
    max_val = config['training']['max_val']
    # min_val = np.min(data)
    # max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)

    # min_val = np.min(data, axis=(0, 1))
    # max_val = np.max(data, axis=(0, 1))
    # min_val = np.expand_dims(min_val, axis=(0, 1))
    # max_val = np.expand_dims(max_val, axis=(0, 1))
    # normalized_data = np.nan_to_num((data - min_val) / (max_val - min_val + 1e-6))
    # numerator = data - min_val
    # denominator = max_val - min_val
    # normalized_data = np.where(denominator != 0, numerator / denominator, 0)

    # sclar = MinMaxScaler(feature_range=(0, 1))
    # normalized_data = sclar.fit_transform(data)
    return normalized_data


def z_score_standardization(data):
    """
    对三维数据进行标准化，假设数据形状为 (batch_size, time_steps, features)。
    """
    # 初始化一个数组来保存标准化后的数据
    normalized_data = np.zeros_like(data)
    
    # 对每个批次的数据进行遍历
    for i in range(data.shape[0]):
        # 对每个时间步的特征进行遍历
        for j in range(data.shape[1]):
            # 计算每个特征的平均值和标准差
            mean = np.mean(data[i, :, :], axis=0)
            std = np.std(data[i, :, :], axis=0)
            
            # 对每个特征进行标准化
            normalized_data[i, j, :] = (data[i, j, :] - mean) / (std + 1e-6)
    
    return normalized_data


def plot_confusion_matrix(y_true, y_pred, labels, save_path='img/confusion_matrix.png', title='Confusion matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵图

    参数:
    y_true -- 真实标签
    y_pred -- 预测标签
    labels -- 分类标签列表
    title -- 图的标题
    cmap -- 颜色映射
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 绘制图形
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    # 绘制标签
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    
    # 绘制混淆矩阵中的值
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.tight_layout()
    plt.ylabel('ground-truth')
    plt.xlabel('prediction')
    # plt.show()
    plt.savefig(save_path, dpi=300)


def mean_filter(signal, window_size=3):
    """
    对二维信号数组进行均值滤波
    :param signal: 二维信号数组，第一维是时间，第二维是特征大小
    :param window_size: 滤波窗口大小
    :return: 滤波后的信号数组
    """
    # 对信号进行边缘扩展，以处理窗口边缘的情况
    pad_width = window_size // 2
    pad_signal = np.pad(signal, ((pad_width, pad_width), (0, 0)), mode='edge')
    
    # 初始化滤波后的信号数组
    filtered_signal = np.zeros_like(signal)
    
    # 对信号进行均值滤波
    for i in range(signal.shape[0]):
        filtered_signal[i] = np.mean(pad_signal[i:i+window_size], axis=0)
    
    return filtered_signal


def median_filter(signal, window_size=3):
    """
    对二维信号数组进行中值滤波
    :param signal: 二维信号数组，第一维是时间，第二维是特征大小
    :param window_size: 滤波窗口大小
    :return: 滤波后的信号数组
    """
    # 对信号进行边缘扩展，以处理窗口边缘的情况
    pad_width = window_size // 2
    pad_signal = np.pad(signal, ((pad_width, pad_width), (0, 0)), mode='edge')
    
    # 初始化滤波后的信号数组
    filtered_signal = np.zeros_like(signal)
    
    # 对信号进行均值滤波
    for i in range(signal.shape[0]):
        filtered_signal[i] = np.median(pad_signal[i:i+window_size], axis=0)
    
    return filtered_signal


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "same") / w


def load_csv_data(data_dir):
    config = load_config('config.yaml')
    window_size = config['data']['window_size']
    window_step = config['data']['window_step']
    window_data = []
    window_label = []
    data_files = os.listdir(data_dir)
    for data_file in data_files:
        if data_file.endswith('.csv'):
            fileNum = data_file.split('_')[0]
            class_label = fileNum2classes[fileNum]
            csv_path = os.path.join(data_dir, data_file)
            data = pd.read_csv(csv_path, on_bad_lines='skip', skiprows=[0])
            data = data.reset_index(drop=True)
            columns = list(data.columns)
            columns2 = []
            for i in range(len(columns)):
                columns2.append(columns[i].strip())
            for i in range(len(data.columns)):
                data.rename(columns={data.columns[i]: columns2[i]}, inplace=True)
            Line_value = np.array(data['Line'])
            Line_value = sorted(np.unique(Line_value))
            for l in Line_value:
                df = data.loc[data['Line'] == l]
                features = df[feature_list]
                features = np.array(features)
                # 滤波
                # features = mean_filter(features, window_size=3)  # 均值滤波
                features = median_filter(features, window_size=3)  # 中值滤波
                index = 0
                while (index * window_step + window_size) <= features.shape[0]:
                    start = index * window_step
                    end = start + window_size
                    window = features[start:end]
                    window_data.append(window)
                    window_label.append(class_label)
                    index += 1
    window_data = np.array(window_data)
    window_label = np.array(window_label)
    return window_data, window_label


def label_to_one_hot(label, num_classes):
    one_hot = torch.zeros(len(label), num_classes)
    one_hot.scatter_(1, label.view(-1, 1), 1)
    return one_hot


def find_data_belong(sample):
    """
    sample: 10 * window_size的numpy数组
    """
    config = load_config('config.yaml')
    window_size = config['data']['window_size']
    window_step = config['data']['window_step']
    raw_data_dir = '../data/确认场地'
    nums = ['120', '122', '123', '124', '125']
    for n in nums:
        csv_path = os.path.join(raw_data_dir, '{}_G726_em+inv.csv'.format(n))
        data = pd.read_csv(csv_path, on_bad_lines='skip', skiprows=[0])
        data = data.reset_index(drop=True)
        columns = list(data.columns)
        columns2 = []
        for i in range(len(columns)):
            columns2.append(columns[i].strip())
        for i in range(len(data.columns)):
            data.rename(columns={data.columns[i]: columns2[i]}, inplace=True)
        
        Line_value = np.array(data['Line'])
        Line_value = sorted(np.unique(Line_value))
        for l in Line_value:
            df = data.loc[data['Line'] == l]
            features = df[feature_list]
            features = np.array(features)
            index = 0
            while (index * window_step + window_size) <= features.shape[0]:
                start = index * window_step
                end = start + window_size
                window = features[start:end]
                if (sample == window).any():
                    print(f"In {csv_path}, line {l}, start index {start}, end index {end}")
                index += 1

        # npy_path = os.path.join(raw_data_dir, 'processed_data', '{}_G726_em+inv_data.npy'.format(n))
        # dataset = np.load(npy_path)


def load_dataset_random_split(data_path, label_path):
    config = load_config('config.yaml')
    train_ratio = config['training']['train_ratio']
    # 加载数据集
    data = np.load(data_path)
    label = np.load(label_path)
    # data, label = load_csv_data('../data/20240409正交实验/raw_data')

    # 归一化数据到 [0, 1] 范围
    data = normalize(data)
    # z-score标准化
    # data = z_score_standardization(data)

    # data_mean = np.mean(normalized_data, axis=(0, 1))
    # data_std = np.std(normalized_data, axis=(0, 1))
    # mean = np.mean(data_mean)
    # std = np.mean(data_std)
    # 数据预处理
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=data_mean, std=data_std)])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(mean,), std=(std,))])
    transform = None

    uxo_dataset = UXODataset(data, label, transform=transform)
    length = len(uxo_dataset)
    train_size, val_size = int(train_ratio * length), length-int(train_ratio * length)
    train_dataset, test_dataset = torch.utils.data.random_split(uxo_dataset, [train_size, val_size])
    return train_dataset, test_dataset


def load_dataset(train_data_path, train_label_path, test_data_path, test_label_path):
    config = load_config('config.yaml')
    train_ratio = config['training']['train_ratio']
    train_data = np.load(train_data_path)
    train_label = np.load(train_label_path)
    test_data = np.load(test_data_path)
    test_label = np.load(test_label_path)

    train_data = normalize(train_data)
    test_data = normalize(test_data)
    transform = None
    train_dataset = UXODataset(train_data, train_label, transform=transform)
    test_dataset = UXODataset(test_data, test_label, transform=transform)
    return train_dataset, test_dataset


def read_process_csv(csv_path):
    # 读取导出的csv数据，去掉开头第一行
    data = pd.read_csv(csv_path, on_bad_lines='skip', skiprows=[0])
    data = data.reset_index(drop=True)

    # 去掉列索引中的空格
    columns = list(data.columns)
    # print(columns)
    columns2 = []
    for i in range(len(columns)):
        columns2.append(columns[i].strip())

    for i in range(len(data.columns)):
        data.rename(columns={data.columns[i]: columns2[i]}, inplace=True)
    return data


if __name__ == "__main__":
    data, label = load_csv_data('../data/20240409正交实验/raw_data')
    print(data.shape, label.shape)