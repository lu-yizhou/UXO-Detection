import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
import torch
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import LSTMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import *
from process_raw_data import fileNum2setting


def get_test_metric(csv_path, model, config):
    window_size = config['data']['window_size']
    window_step = config['data']['window_step']
    data = read_process_csv(csv_path)
    with open("line_dataset.json", 'r') as f:
        line_dataset = json.load(f)
    test_line = line_dataset[csv_path]['test']
    acc_list = []
    prec_list = []
    recall_list = []
    f1_list = []
    I_425Hz_max_list = []
    Q_425Hz_max_list = []
    I_1525Hz_max_list = []
    Q_1525Hz_max_list = []
    I_5325Hz_max_list = []
    I_18325Hz_max_list = []
    I_63025Hz_max_list = []
    diff_list = []
    for l in test_line:
        print(f'line: {l}')
        df = data.loc[data['Line'] == l]
        features = df[feature_list]
        features = np.array(features)
        I_425Hz = features[:, 0]
        Q_425Hz = features[:, 1]
        I_1525Hz = features[:, 2]
        Q_1525Hz = features[:, 3]
        I_5325Hz = features[:, 4]
        I_18325Hz = features[:, 6]
        I_63025Hz = features[:, 8]
        I_425Hz_max_list.append(np.max(I_425Hz))
        Q_425Hz_max_list.append(np.max(Q_425Hz))
        I_1525Hz_max_list.append(np.max(I_1525Hz))
        Q_1525Hz_max_list.append(np.max(Q_1525Hz))
        I_5325Hz_max_list.append(np.max(I_5325Hz))
        I_18325Hz_max_list.append(np.max(I_18325Hz))
        I_63025Hz_max_list.append(np.max(I_63025Hz))
        diff_value = np.max(I_425Hz) - np.min(I_425Hz)
        diff_list.append(diff_value)
        index = 0
        window_data = []
        window_label = []
        while (index * window_step + window_size) <= features.shape[0]:
            start = index * window_step
            end = start + window_size
            window = features[start:end]
            df_label = df['label']
            w_l = np.array(df_label.iloc[start:end], dtype=int)
            counts = np.bincount(w_l)
            # 出现次数最多的元素
            most_common_label = np.argmax(counts)
            if most_common_label == 2 and (np.max(counts) / window_size > 2/3):
                w_label = 2
            elif most_common_label == 1 and (np.max(counts) / window_size > 2/3):
                w_label = 1
            else:
                w_label = 0
            window_label.append(w_label)
            window_data.append(window)
            index += 1
        window_data = np.array(window_data)
        window_data = normalize(window_data)
        uxo_dataset = UXODataset(window_data, window_label)
        data_loader = DataLoader(uxo_dataset, batch_size=1, shuffle=False)
        y_pred = []
        y_pred_prob = []
        y_true = []
        with torch.no_grad():
            for w, target in data_loader:
                w = w.to('cuda')
                out = model(w)
                probability = F.softmax(out, dim=1)
                probability = probability.view(-1).detach().cpu().numpy()
                y_pred_prob.extend(probability)
                pred = out.argmax(dim=1, keepdim=True)
                pred = pred.view(-1).detach().cpu().numpy()
                target = target.numpy()
                y_pred.extend(pred)
                y_true.extend(target)
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_pred_prob = np.array(y_pred_prob)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        if accuracy == 0 or precision == 0 or recall == 0 or f1 == 0:
            continue
        else:
            acc_list.append(accuracy)
            prec_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1 score: {f1}")
    return acc_list, prec_list, recall_list, f1_list, I_425Hz_max_list, Q_425Hz_max_list, I_1525Hz_max_list, Q_1525Hz_max_list, I_5325Hz_max_list, I_18325Hz_max_list, I_63025Hz_max_list, diff_list


def get_mean_metric(csv_path, config, model):
    window_size = config['data']['window_size']
    window_step = config['data']['window_step']
    data = read_process_csv(csv_path)
    with open("line_dataset.json", 'r') as f:
        line_dataset = json.load(f)
    test_line = line_dataset[csv_path]['test']
    window_data = []
    window_label = []
    for l in test_line:
        df = data.loc[data['Line'] == l]
        features = df[feature_list]
        features = np.array(features)
        index = 0
        while (index * window_step + window_size) <= features.shape[0]:
            start = index * window_step
            end = start + window_size
            window = features[start:end]
            df_label = df['label']
            w_l = np.array(df_label.iloc[start:end], dtype=int)
            counts = np.bincount(w_l)
            # 出现次数最多的元素
            most_common_label = np.argmax(counts)
            if most_common_label == 2 and (np.max(counts) / window_size > 2/3):
                w_label = 2
            elif most_common_label == 1 and (np.max(counts) / window_size > 2/3):
                w_label = 1
            else:
                w_label = 0
            window_label.append(w_label)
            window_data.append(window)
            index += 1
    window_data = np.array(window_data)
    window_data = normalize(window_data)
    uxo_dataset = UXODataset(window_data, window_label)
    data_loader = DataLoader(uxo_dataset, batch_size=1, shuffle=False)
    y_pred = []
    y_pred_prob = []
    y_true = []
    with torch.no_grad():
        for w, target in data_loader:
            w = w.to('cuda')
            out = model(w)
            probability = F.softmax(out, dim=1)
            probability = probability.view(-1).detach().cpu().numpy()
            y_pred_prob.append(probability[1])
            pred = out.argmax(dim=1, keepdim=True)
            pred = pred.view(-1).detach().cpu().numpy()
            target = target.numpy()
            y_pred.extend(pred)
            y_true.extend(target)
            # print(target, pred, probability)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1 score: {f1}")
    return accuracy, precision, recall, f1



def get_num2metric(model, config):
    num2metric = {}
    csv_file = {
        "文件编号": [],
        "埋深(m)": [],
        "高度(m)": [],
        "角度(°)": [],
        "准确率(accuracy)": [],
        "精确率(precision)": [],
        "召回率(recall)": [],
        "F1分数": [],
        "极大值": [],
        "极差": []
    }
    nums = ['120', '122', '123', '125', '127', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140']
    for n in nums:
        csv_path = f"../data/确认场地/{n}_G726_em+inv.csv"
        print(csv_path)
        acc_list, prec_list, recall_list, f1_list, I_425Hz_max_list, Q_425Hz_max_list, I_1525Hz_max_list, Q_1525Hz_max_list, I_5325Hz_max_list, I_18325Hz_max_list, I_63025Hz_max_list, diff_list = get_test_metric(csv_path, model, config)
        num2metric[n] = {
            "accuracy": acc_list,
            "precision": prec_list,
            "recall": recall_list,
            "f1 score": f1_list,
            "I_425Hz_max": I_425Hz_max_list,
            "Q_425Hz_max": Q_425Hz_max_list,
            "I_1525Hz_max": I_1525Hz_max_list,
            "Q_1525Hz_max": Q_1525Hz_max_list,
            "I_5325Hz_max": I_5325Hz_max_list,
            "I_18325Hz_max": I_18325Hz_max_list,
            "I_63025Hz_max": I_63025Hz_max_list,
            "diff": diff_list
        }
        csv_file['文件编号'].append(n)
        [_, depth, height, degree, _] = fileNum2setting[n]
        csv_file['埋深(m)'].append(depth)
        csv_file['高度(m)'].append(height)
        csv_file['角度(°)'].append(degree)
        csv_file['准确率(accuracy)'].append(np.mean(acc_list))
        csv_file['精确率(precision)'].append(np.mean(prec_list))
        csv_file['召回率(recall)'].append(np.mean(recall_list))
        csv_file['F1分数'].append(np.mean(f1_list))
        csv_file['极大值'].append(np.mean(I_425Hz_max_list))
        csv_file['极差'].append(np.mean(diff_list))
    csv_file = pd.DataFrame(csv_file)
    csv_file.to_csv('static_result.csv')
    return num2metric


def num2metric_anova(num2metric):
    data = {
        "num": [],
        "depth": [],
        "height": [],
        "degree": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "I_425Hz_max": [],
        "Q_425Hz_max": [],
        "I_1525Hz_max": [],
        "Q_1525Hz_max": [],
        "I_5325Hz_max": [],
        "I_18325Hz_max": [],
        "I_63025Hz_max": [],
        "diff": []
    }
    for n, metrics in num2metric.items():
        [_, depth, height, degree, _] = fileNum2setting[n]
        acc_list = metrics['accuracy']
        prec_list = metrics['precision']
        recall_list = metrics['recall']
        f1_list = metrics['f1 score']
        I_425Hz_max = metrics['I_425Hz_max']
        Q_425Hz_max = metrics['Q_425Hz_max']
        I_1525Hz_max = metrics['I_1525Hz_max']
        Q_1525Hz_max = metrics['Q_1525Hz_max']
        I_5325Hz_max = metrics['I_5325Hz_max']
        I_18325Hz_max = metrics['I_18325Hz_max']
        I_63025Hz_max = metrics['I_63025Hz_max']
        diff_list = metrics['diff']
        for (acc, p, r, f, I_425Hz, Q_425Hz, I_1525Hz, Q_1525Hz, I_5325Hz, I_18325Hz, I_63025Hz, d) in zip(acc_list, prec_list, recall_list, f1_list, I_425Hz_max, Q_425Hz_max, I_1525Hz_max, Q_1525Hz_max, I_5325Hz_max, I_18325Hz_max, I_63025Hz_max, diff_list):
            data['num'].append(n)
            data['depth'].append(depth)
            data['height'].append(height)
            data['degree'].append(degree)
            data['accuracy'].append(acc)
            data['precision'].append(p)
            data['recall'].append(r)
            data['f1'].append(f)
            data['I_425Hz_max'].append(I_425Hz)
            data['Q_425Hz_max'].append(Q_425Hz)
            data['I_1525Hz_max'].append(I_1525Hz)
            data['Q_1525Hz_max'].append(Q_1525Hz)
            data['I_5325Hz_max'].append(I_5325Hz)
            data['I_18325Hz_max'].append(I_18325Hz)
            data['I_63025Hz_max'].append(I_63025Hz)
            data['diff'].append(d)
    
    data = pd.DataFrame(data)
    
    # 设置ANOVA模型
    model = ols('f1 ~ C(depth) + C(height) + C(degree)', data=data).fit()
    # model = ols('max ~ C(depth)', data=data).fit()

    # 进行ANOVA
    anova_results = sm.stats.anova_lm(model, typ=2)

    # 打印ANOVA结果
    print(anova_results)


def num2metric_multi_anova(num2metric, logger=None):
    data = {
        "num": [],
        "depth": [],
        "height": [],
        "degree": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "I_425Hz_max": [],
        "Q_425Hz_max": [],
        "I_1525Hz_max": [],
        "Q_1525Hz_max": [],
        "I_5325Hz_max": [],
        "I_18325Hz_max": [],
        "I_63025Hz_max": [],
        "diff": []
    }
    for n, metrics in num2metric.items():
        [_, depth, height, degree, _] = fileNum2setting[n]
        acc_list = metrics['accuracy']
        prec_list = metrics['precision']
        recall_list = metrics['recall']
        f1_list = metrics['f1 score']
        I_425Hz_max = metrics['I_425Hz_max']
        Q_425Hz_max = metrics['Q_425Hz_max']
        I_1525Hz_max = metrics['I_1525Hz_max']
        Q_1525Hz_max = metrics['Q_1525Hz_max']
        I_5325Hz_max = metrics['I_5325Hz_max']
        I_18325Hz_max = metrics['I_18325Hz_max']
        I_63025Hz_max = metrics['I_63025Hz_max']
        diff_list = metrics['diff']
        for (acc, p, r, f, I_425Hz, Q_425Hz, I_1525Hz, Q_1525Hz, I_5325Hz, I_18325Hz, I_63025Hz, d) in zip(acc_list, prec_list, recall_list, f1_list, I_425Hz_max, Q_425Hz_max, I_1525Hz_max, Q_1525Hz_max, I_5325Hz_max, I_18325Hz_max, I_63025Hz_max, diff_list):
            data['num'].append(n)
            data['depth'].append(depth)
            data['height'].append(height)
            data['degree'].append(degree)
            data['accuracy'].append(acc)
            data['precision'].append(p)
            data['recall'].append(r)
            data['f1'].append(f)
            data['I_425Hz_max'].append(I_425Hz)
            data['Q_425Hz_max'].append(Q_425Hz)
            data['I_1525Hz_max'].append(I_1525Hz)
            data['Q_1525Hz_max'].append(Q_1525Hz)
            data['I_5325Hz_max'].append(I_5325Hz)
            data['I_18325Hz_max'].append(I_18325Hz)
            data['I_63025Hz_max'].append(I_63025Hz)
            data['diff'].append(d)
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(data)

    # 设置MANOVA的模型
    # manova = MANOVA.from_formula('I_425Hz_max + Q_425Hz_max + I_1525Hz_max + Q_1525Hz_max + I_5325Hz_max + I_18325Hz_max + I_63025Hz_max ~ C(depth) + C(height) + C(degree)', data=df)
    manova = MANOVA.from_formula('accuracy + precision + recall + f1 ~ C(depth) + C(height) + C(degree)', data=df)

    # 进行MANOVA
    result = manova.mv_test()
    if logger:
        logger.info('MANOVA result:')
        logger.info(result.summary())
    else:
        print('MANOVA result:')
        print(result.summary())    


if __name__ == "__main__":
    config = load_config('config.yaml')
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    num_classes = config['model']['num_classes']
    
    seed = config['training']['seed']
    train_ratio = config['training']['train_ratio']
    test_batch_size = config['training']['test_batch_size']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LSTMClassifier(input_size=len(feature_list), hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
    weight = torch.load('./checkpoint/ckpt_96.29.pth')
    model.load_state_dict(weight['model'])
    model = model.to(device)
    model.eval()

    num2metric = get_num2metric(model, config)
    num2metric_multi_anova(num2metric)

