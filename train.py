import os
import logging
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import LSTMClassifier
from dataloader import UXODataset
from utils import *
from anova import get_num2metric, num2metric_multi_anova, mean_metric_f_oneway


start_time = time.time()

# 日志设置
logger = logging.getLogger('common')
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
log_file = 'log/{}.log'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
th = logging.FileHandler(log_file, encoding="utf-8")
logger.addHandler(sh)
logger.addHandler(th)

config = load_config('config.yaml')
log_config(config, logger)

hidden_size = config['model']['hidden_size']
num_layers = config['model']['num_layers']
num_classes = config['model']['num_classes']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
train_batch_size = config['training']['train_batch_size']
test_batch_size = config['training']['test_batch_size']
epoch_num = config['training']['epoch_num']
train_ratio = config['training']['train_ratio']
data_path = config['data']['data_path']
label_path = config['data']['label_path']
train_data_path = config['data']['train_data_path']
train_label_path = config['data']['train_label_path']
test_data_path = config['data']['test_data_path']
test_label_path = config['data']['test_label_path']


device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info("Using device: {}".format(device))

# 实例化网络
# model = MLP()
model = LSTMClassifier(input_size=10, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
# model = TimeSeriesCNN_1d(in_channels=10, num_classes=num_classes)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(device))
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# train_dataset, test_dataset = load_dataset_random_split(data_path, label_path)
train_dataset, test_dataset = load_dataset(train_data_path, train_label_path, test_data_path, test_label_path)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


# 训练网络
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            logger.info('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, epoch_num, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    return train_loss


# 测试网络
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_true.extend(target.view(-1).detach().cpu().numpy())
            y_pred.extend(pred.view(-1).detach().cpu().numpy())
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    test_loss /= len(test_loader.dataset)
    acc = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Precision: {:.4f}, Recall: {:.4f}, F1 score: {:.4f}\n'.format(test_loss, correct, len(test_loader.dataset), acc, precision, recall, f1))
    if acc > best_acc:
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_path = 'checkpoint/ckpt_{}.pth'.format(round(acc, 2))
        torch.save(state, save_path)
        logger.info('Save model to {} at epoch {}'.format(save_path, epoch))
        best_acc = acc
    return test_loss, acc
    

def get_confusion_matrix():
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
    # labels = ["background", "clutter", "UXO"]
    labels = ['background', 'UXO']
    plot_confusion_matrix(y_true, y_pred, save_path='img/confusion_matrix.png', labels=labels)


def plot_loss_acc(train_loss_list, test_loss_list, test_acc_list):
    plt.figure()
    plt.plot(train_loss_list, label="train loss")
    plt.plot(test_loss_list, label="test loss")
    plt.legend()
    plt.savefig("img/loss.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(test_acc_list, label="accuracy")
    plt.ylim(50, 100)
    plt.legend()
    plt.savefig("img/acc.png", dpi=300)
    plt.close()


best_acc = 0
train_loss_list = []
test_loss_list = []
test_acc_list = []
# 运行训练和测试
for epoch in range(epoch_num):
    train_loss = train(epoch)
    test_loss, acc = test(epoch)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    test_acc_list.append(acc)

logger.info("Best acc: {:.4f}%".format(best_acc))

end_time = time.time()
s = end_time - start_time
h, m, s = seconds2hours(s)
logger.info("Consuming time: {}h{}m{}s".format(h, m, s))
