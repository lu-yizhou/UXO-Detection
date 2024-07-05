import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import json
from utils import *


data_dir = 'data/dir'
config = load_config('config.yaml')
window_size = config['data']['window_size']
window_step = config['data']['window_step']
train_ratio = config['training']['train_ratio']

classes = {
     0: "background",
     1: "clutter",
     2: "UXO"
}

line_dataset = {}

fileNum2setting = {  # file num: [class label, depth(m), height(m), degree, speed]
     "120": [2, 0.3, 0.5, 90, 'walk'],
     "122": [2, 0.3, 1.0, 90, 'walk'],
     "123": [2, 0.3, 1.0, 60, 'walk'],
     "124": [2, 0.3, 1.5, 45, 'walk'],
     "125": [2, 0.3, 2.0, 45, 'walk'],
     "127": [2, 0.3, 1.5, 45, 'walk'],
     "131": [2, 0.1, 0.5, 45, 'walk'],
     "132": [2, 0.1, 1.5, 45, 'walk'],
     "133": [2, 0.1, 1.0, 90, 'walk'],
     "134": [2, 0.1, 0.5, 60, 'walk'],
     "135": [2, 0.1, 2.0, 60, 'walk'],
     "136": [2, 0.6, 1.0, 45, 'walk'],
     "137": [2, 0.6, 0.5, 60, 'walk'],
     "138": [2, 0.6, 1.5, 60, 'walk'],
     "139": [2, 0.6, 1.5, 90, 'walk'],
     "140": [2, 0.6, 2.0, 90, 'walk']
}


def raw_csv_2_npy(csv_path, save_dir, label=0):
     """
     csv_path: csv文件路径
     save_dir: 保存数据与标签的文件夹路径
     label: 标签
     """
     file_name = os.path.splitext(os.path.split(csv_path)[1])[0]
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

     # fig = plt.figure()
     # plt.plot(data[feature_list])
     # plt.savefig('img/{}.png'.format(file_name))

     window_data = []
     window_label = []
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
               if 'label' in columns:
                    df_label = df['label']
                    w_l = np.array(df_label.iloc[start:end], dtype=int)
                    counts = np.bincount(w_l)
                    # 出现次数最多的元素
                    most_common_label = np.argmax(counts)
                    if most_common_label == 2 and (np.max(counts) / window_size > 0.8):
                         w_label = 2
                    elif most_common_label == 1 and (np.max(counts) / window_size > 0.8):
                         w_label = 1
                    else:
                         w_label = 0
                    window_label.append(w_label)
                    window_data.append(window)
               else:
                    window_data.append(window)
                    window_label.append(label)
               index += 1
     window_data = np.array(window_data)
     window_label = np.array(window_label)
     # save_path = os.path.join(data_dir, '{}_{}'.format(window_size, window_step))
     if not os.path.exists(save_dir):
          os.mkdir(save_dir)
     np.save(os.path.join(save_dir, '{}_data.npy'.format(file_name)), window_data)
     np.save(os.path.join(save_dir, '{}_label.npy'.format(file_name)), window_label)
     print(f"Transfer finished. data shape: {window_data.shape}, label shape: {window_label.shape}")
     print("Number of label 0: {}, 1: {}, 2: {}".format(np.sum(window_label == 0), np.sum(window_label == 1), np.sum(window_label == 2)))


def merge_data(data_dir):
     all_data = []
     all_label = []
     file_nums = list(fileNum2setting.keys())
     files = os.listdir(data_dir)
     for num in file_nums:
          data_file = f"{num}_G726_em+inv_data.npy"
          label_file = f"{num}_G726_em+inv_label.npy"
          if data_file in files and label_file in files:
               data = np.load(os.path.join(data_dir, data_file))
               label = np.load(os.path.join(data_dir, label_file))
               all_data.extend(data)
               all_label.extend(label)
               print(f'Add {data_file} and {label_file}')
     all_data = np.array(all_data)
     all_label = np.array(all_label)
     np.save(os.path.join(data_dir, 'all_data.npy'), all_data)
     np.save(os.path.join(data_dir, 'all_label.npy'), all_label)
     print(f"Merge finished. data shape: {all_data.shape}, label shape: {all_label.shape}")
     print("Number of label 0: {}, 1: {}, 2: {}".format(np.sum(all_label == 0), np.sum(all_label == 1), np.sum(all_label == 2)))


def plot_curves(csv_path):
     file_name = os.path.splitext(os.path.split(csv_path)[1])[0]
     data = pd.read_csv(csv_path, on_bad_lines='skip', skiprows=[0])
     data = data.reset_index(drop=True)
     columns = list(data.columns)
     print(columns)
     columns2 = []
     for i in range(len(columns)):
          columns2.append(columns[i].strip())

     for i in range(len(data.columns)):
          data.rename(columns={data.columns[i]: columns2[i]}, inplace=True)
     fig = plt.figure()
     plt.plot(data[feature_list])
     # plt.show()
     plt.savefig('img/{}.png'.format(file_name))


def process_all_raw_data(raw_data_dir, save_dir):
     files = os.listdir(raw_data_dir)
     for file_name in files:
          if file_name.endswith('.csv'):
               name, ext = os.path.splitext(file_name)
               file_num = name.split('_')[0]
               label = fileNum2setting[file_num][0]
               file_path = os.path.join(raw_data_dir, file_name)
               raw_csv_2_npy(file_path, save_dir, label)


def split_dataset_from_csv(csv_path, dataset_dir, line_dataset):
     file_name = os.path.splitext(os.path.split(csv_path)[1])[0]
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

     train_lines = line_dataset[csv_path]['train']
     test_lines = line_dataset[csv_path]['test']

     window_data = []
     window_label = []
     for l in train_lines:
          df = data.loc[data['Line'] == l]
          features = df[feature_list]
          features = np.array(features)
          index = 0
          while (index * window_step + window_size) <= features.shape[0]:
               start = index * window_step
               end = start + window_size
               window = features[start:end]
               if 'label' in columns:
                    df_label = df['label']
                    w_l = np.array(df_label.iloc[start:end], dtype=int)
                    counts = np.bincount(w_l)
                    # 出现次数最多的元素
                    most_common_label = np.argmax(counts)
                    if most_common_label == 2 and (np.max(counts) / window_size > 0.8):
                         w_label = 2
                    elif most_common_label == 1 and (np.max(counts) / window_size > 0.8):
                         w_label = 1
                    else:
                         w_label = 0
                    window_label.append(w_label)
                    window_data.append(window)
               else:
                    window_data.append(window)
                    window_label.append(label)
               index += 1
     window_data = np.array(window_data)
     window_label = np.array(window_label)
     
     if not os.path.exists(dataset_dir):
          os.mkdir(dataset_dir)
     save_dir = os.path.join(dataset_dir, 'train')
     if not os.path.exists(save_dir):
          os.mkdir(save_dir)
     np.save(os.path.join(save_dir, '{}_data.npy'.format(file_name)), window_data)
     np.save(os.path.join(save_dir, '{}_label.npy'.format(file_name)), window_label)

     window_data = []
     window_label = []
     for l in test_lines:
          df = data.loc[data['Line'] == l]
          features = df[feature_list]
          features = np.array(features)
          index = 0
          while (index * window_step + window_size) <= features.shape[0]:
               start = index * window_step
               end = start + window_size
               window = features[start:end]
               if 'label' in columns:
                    df_label = df['label']
                    w_l = np.array(df_label.iloc[start:end], dtype=int)
                    counts = np.bincount(w_l)
                    # 出现次数最多的元素
                    most_common_label = np.argmax(counts)
                    if most_common_label == 2 and (np.max(counts) / window_size > 0.8):
                         w_label = 2
                    elif most_common_label == 1 and (np.max(counts) / window_size > 0.8):
                         w_label = 1
                    else:
                         w_label = 0
                    window_label.append(w_label)
                    window_data.append(window)
               else:
                    window_data.append(window)
                    window_label.append(label)
               index += 1
     window_data = np.array(window_data)
     window_label = np.array(window_label)
     
     if not os.path.exists(dataset_dir):
          os.mkdir(dataset_dir)
     save_dir = os.path.join(dataset_dir, 'test')
     if not os.path.exists(save_dir):
          os.mkdir(save_dir)
     np.save(os.path.join(save_dir, '{}_data.npy'.format(file_name)), window_data)
     np.save(os.path.join(save_dir, '{}_label.npy'.format(file_name)), window_label)


def get_line_dataset():
     line_dataset = {}
     nums = ['120', '122', '123', '125', '127', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140']
     for n in nums:
          csv_path = os.path.join(raw_data_dir, '{}_G726_em+inv.csv'.format(n))
          data = read_process_csv(csv_path)
          Line_value = np.array(data['Line'])
          Line_value = sorted(np.unique(Line_value))
          np.random.shuffle(Line_value)
          train_size = int(round(len(Line_value) * train_ratio))
          test_size = len(Line_value) - train_size
          print(csv_path)
          print(f"train line number: {Line_value[:train_size]}")
          print(f"test line number: {Line_value[train_size:]}")
          line_dataset[csv_path] = {
               "train": [int(i) for i in Line_value[:train_size]],
               "test": [int(i) for i in Line_value[train_size:]]
          }
     return line_dataset


if __name__ == "__main__":
     raw_data_dir = 'data/dir'
     line_dataset = get_line_dataset()
     dataset_dir = 'path/to/dataset'
     nums = ['120', '122', '123', '125', '127', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140']
     for n in nums:
          csv_path = os.path.join(raw_data_dir, '{}_G726_em+inv.csv'.format(n))
          split_dataset_from_csv(csv_path, dataset_dir, line_dataset)
     
     merge_data(os.path.join(dataset_dir, 'train'))
     merge_data(os.path.join(dataset_dir, 'test'))
     