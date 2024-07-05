import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import *
from model import MLP, LSTMClassifier
from dataloader import UXODataset


config = load_config('config.yaml')

hidden_size = config['model']['hidden_size']
num_layers = config['model']['num_layers']
num_classes = config['model']['num_classes']
test_batch_size = config['training']['test_batch_size']
train_ratio = config['training']['train_ratio']
window_size = config['data']['window_size']
window_step = config['data']['window_step']
data_path = config['data']['data_path']
label_path = config['data']['label_path']
train_data_path = config['data']['train_data_path']
train_label_path = config['data']['train_label_path']
test_data_path = config['data']['test_data_path']
test_label_path = config['data']['test_label_path']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset, test_dataset = load_dataset(train_data_path, train_label_path, test_data_path, test_label_path)

test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


model = LSTMClassifier(input_size=len(feature_list), hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
weight = torch.load('./checkpoint/ckpt_96.29.pth')
model.load_state_dict(weight['model'])
model = model.to(device)
model.eval()


def predict_signal(data):
    window_step = 1
    data = np.array(data, dtype=np.float32)
    windows = []
    index = 0
    while (index * window_step + window_size) <= data.shape[0]:
        start = index * window_step
        end = start + window_size
        window = data[start:end]
        windows.append(window)
        index += 1
    windows = np.array(windows)
    windows = normalize(windows)
    uxo_dataset = UXODataset(windows)
    data_loader = DataLoader(uxo_dataset, batch_size=test_batch_size, shuffle=False)
    output = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            probability = F.softmax(out, dim=1)[:, 1]
            # output.extend(pred)
            output.extend(probability)
    output = torch.stack(output)
    return output


def predict_csv(csv_path):
    file_name = os.path.splitext(os.path.split(csv_path)[1])[0]
    # 读取导出的csv数据，去掉开头第一行
    data = pd.read_csv(csv_path, on_bad_lines='skip', skiprows=[0])
    data = data.reset_index(drop=True)

    # 去掉列索引中的空格
    columns = list(data.columns)
    columns2 = []
    for i in range(len(columns)):
        columns2.append(columns[i].strip())

    for i in range(len(data.columns)):
        data.rename(columns={data.columns[i]: columns2[i]}, inplace=True)

    Line_value = np.array(data['Line'])
    Line_value = sorted(np.unique(Line_value))
    out = []
    for l in Line_value:
        df = data.loc[data['Line'] == l]
        features = df[feature_list]
        features = np.array(features)
        prediction = predict_signal(features)
        prediction = prediction.detach().cpu().numpy()
        out.append(prediction)
    return out


def predict_csv_line(csv_path, line_value):
    data = read_process_csv(csv_path)
    df = data.loc[data['Line'] == line_value]
    features = df[feature_list]
    features = np.array(features)
    padded_features = np.pad(features, ((int(window_size/2), int(window_size/2)), (0, 0)), mode='constant', constant_values=0)
    prediction = predict_signal(padded_features)
    prediction = prediction.detach().cpu().numpy().reshape(-1)
    return prediction


def get_confusion_matrix(test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            y_true.extend(target)
            y_pred.extend(pred)
    y_true = torch.stack(y_true).view(-1).detach().cpu().numpy()
    y_pred = torch.stack(y_pred).view(-1).detach().cpu().numpy()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1 score: {f1}")
    labels = ['background', 'UXO']
    plot_confusion_matrix(y_true, y_pred, save_path='img/predict_confusion_matrix.png', labels=labels)


def validate_from_npy(model, data_path, label_path):
    model.eval()
    test_data = np.load(data_path)
    test_label = np.load(label_path)
    test_data = normalize(test_data)
    test_dataset = UXODataset(test_data, test_label, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            y_true.extend(target)
            y_pred.extend(pred)
    y_true = torch.stack(y_true).view(-1).detach().cpu().numpy()
    y_pred = torch.stack(y_pred).view(-1).detach().cpu().numpy()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"data: {data_path}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1 score: {f1}")



prediction = predict_csv('path/to/csv_file.csv')
print(prediction)

