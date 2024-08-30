from __future__ import print_function
import math

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import metrics
from torchvision import transforms
from mlxtend.plotting import plot_confusion_matrix


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 移除额外的标签
def drop_extra_label(df_train, df_test, labels):
    for label in labels:
        df_train.drop(label, axis=1, inplace=True)
        df_test.drop(label, axis=1, inplace=True)

    return pd.concat([df_train, df_test], axis=0)


# 对于离散型特征采用最大最小归一化
def min_max_norm(df, name):
    x = df[name].values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[name] = x_scaled


def log_norm(df, name):
    x = df[name].values.reshape(-1, 1)
    df[name] = np.log10(1 + x)


# 对数据集进行预处理
def data_preprocess_unsw(df):
    # 将proto、state、service、label移到最后几列
    traincols = list(df.columns.values)
    traincols.pop(traincols.index('proto'))
    traincols.pop(traincols.index('state'))
    traincols.pop(traincols.index('service'))
    df = df[traincols + ['proto', 'state', 'service']]

    for i in range(0, len(df.columns.values) - 3):
        if np.max(df[df.columns.values[i]]) < 10:
            min_max_norm(df, df.columns.values[i])
        else:
            log_norm(df, df.columns.values[i])

    # 将所有字符型特征进行onehot encoding
    return pd.get_dummies(df, columns=['proto', 'state', 'service'])


def random_flip(images):
    transform = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomCrop(size=(12, 12)),
        transforms.Resize(size=(14, 14))
    )
    images = transform(images)

    return images


def random_shuffle(images):
    transformed_images = torch.zeros_like(images)
    for index in range(images.size(0)):
        image = np.reshape(images.cpu().numpy()[index], 196)
        np.random.shuffle(image)
        transformed_images[index] = torch.from_numpy(np.reshape(image, (1, 14, 14)))

    transform = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5)
    )
    images = transform(transformed_images)

    return images


def add_labels(df):
    df.columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                  "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                  "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
                  "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                  "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
                  "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                  "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                  "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
                  "attack_cat", "level"]

    return df


def preprocess_labels(df):
    df.drop("level", axis=1, inplace=True)
    is_attack = df['attack_cat'].map(lambda a: 0 if a == 'normal' else 1)
    df['attack_cat'] = is_attack

    return df


def preprocess_labels_multi(df, from_kdd=False):

    if not from_kdd:
        df.drop("level", axis=1, inplace=True)
        label_name = "attack_cat"
    else:
        label_name = "label"
    # 多分类 40
    dos_attacks = ["back", "land", "neptune", "smurf", "teardrop", "pod", "apache2", "udpstorm",
                   "processtable", "mailbomb", "worm"]
    r2l_attacks = ["snmpguess", "httptunnel", "named", "xlock", "xsnoop", "sendmail", "ftp_write",
                   "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "snmpgetattack"]
    u2r_attacks = ["sqlattack", "buffer_overflow", "loadmodule", "perl", "rootkit", "xterm", "ps"]
    probe_attacks = ["ipsweep", "nmap", "portsweep", "satan", "saint", "mscan"]
    classes = ["Normal", "Dos", "R2L", "U2R", "Probe"]
    # label2class = {0: 'Dos', 1: 'Normal', 2: 'Probe', 3: 'R2L', 4: 'U2R'}

    def label2attack(row):
        if row[label_name] in dos_attacks:
            return classes[1]
        if row[label_name] in r2l_attacks:
            return classes[2]
        if row[label_name] in u2r_attacks:
            return classes[3]
        if row[label_name] in probe_attacks:
            return classes[4]

        return classes[0]

    df[label_name] = df.apply(label2attack, axis=1)

    return df


def data_preprocess_nsl(df):
    # 添加标签
    df = add_labels(df)
    # 处理多分类标签标签
    df = preprocess_labels_multi(df)
    # df = preprocess_labels(df)

    traincols = list(df.columns.values)
    traincols.pop(traincols.index('protocol_type'))
    traincols.pop(traincols.index('flag'))
    traincols.pop(traincols.index('service'))
    df = df[traincols + ['protocol_type', 'flag', 'service']]

    for i in range(0, len(df.columns.values) - 4):
        if np.max(df[df.columns.values[i]]) < 10:
            min_max_norm(df, df.columns.values[i])
        else:
            log_norm(df, df.columns.values[i])

    # 由于训练和测试数据集中num_outbound_cmds这一列所有值均为0，故删除此列
    df.drop('num_outbound_cmds', axis=1, inplace=True)

    return pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])


def data_preprocess_kdd(df):

    df.dropna(inplace=True)
    # 将label标签变为二分类问题
    # is_attack = df['label'].map(lambda a: 0 if a == 'normal' else 1)
    # df['label'] = is_attack

    df = preprocess_labels_multi(df, from_kdd=True)

    min_max_norm(df, df.columns[0])
    for i in range(4, len(df.columns.values)-1):
        if np.max(df[df.columns.values[i]]) < 10:
            min_max_norm(df, df.columns.values[i])
        else:
            log_norm(df, df.columns.values[i])

    return pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])


def data_preprocess_cic(df):

    # 将infinity替换为Nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)
    # 将label标签变为二分类问题
    # is_attack = df['Label'].map(lambda a: 0 if a == 'BENIGN' else 1)
    # df['Label'] = is_attack

    for i in range(0, len(df.columns.values)-1):
        if np.max(df[df.columns.values[i]]) > 10 and np.min(df[df.columns.values[i]] > -1):
            log_norm(df, df.columns.values[i])
        else:
            min_max_norm(df, df.columns.values[i])

    return df


def data_preprocess_cidds(df):
    # 将ip_src和ip_dst去掉
    df.drop('Date first seen', axis=1, inplace=True)
    df.drop('Src IP Addr', axis=1, inplace=True)
    df.drop('Dst IP Addr', axis=1, inplace=True)

    # 将label转换为0和1
    # is_attack = df['class'].map(lambda a: 0 if a == 'normal' else 1)
    # df['class'] = is_attack

    # 将Bytes这列中的M转换为byte
    M2b = df['Bytes'].map(lambda a: math.floor(float(a[:-1]) * 1048576) if a.endswith('M') else a)
    df['Bytes'] = M2b.astype(np.int)

    features_ids = [0, 2, 3, 4, 5, 6, 8]
    for i in features_ids:
        if np.max(df[df.columns.values[i]]) > 10 and np.min(df[df.columns.values[i]]) > -1:
            log_norm(df, df.columns.values[i])
        else:
            min_max_norm(df, df.columns.values[i])

    return pd.get_dummies(df, columns=['Proto', 'Flags', 'attackType', 'attackID', 'attackDescription'])


def data_preprocess_bot_iot(df, is_binary=False):
    # 将ip_src和ip_dst去掉，避免模型通过学习到ip地址定位攻击
    df.drop('pkSeqID', axis=1, inplace=True)
    df.drop('saddr', axis=1, inplace=True)
    df.drop('daddr', axis=1, inplace=True)

    # 二分类只需要提取attack这个特征，里面标识了是否为攻击类型
    if is_binary:
        df.drop('category', axis=1, inplace=True)

    df.drop('subcategory', axis=1, inplace=True)

    # 将sport和dport中16进制的数转换为10进制
    hex2dec = df['sport'].map(lambda a: int(a, 16) if a.startswith('0x') else a)
    df['sport'] = hex2dec.astype(int)

    hex2dec = df['dport'].map(lambda a: int(a, 16) if a.startswith('0x') else a)
    df['dport'] = hex2dec.astype(int)


    # 获取数值型和字符型特征列名
    categorical_features = df.select_dtypes(exclude=["number"]).columns.tolist()
    # category特征暂时不用one-hot，后续再one-hot，与其他数据集操作保持一致
    if not is_binary:
        categorical_features.remove('category')

    numeric_features = df.select_dtypes(exclude=['object']).columns

    # 由于attack为类别列，已经转换为0和1，所以不需要归一化
    for numeric_feature in numeric_features[:-1]:
        if np.max(df[numeric_feature]) > 10 and np.min(df[numeric_feature]) > -1:
            log_norm(df, numeric_feature)
        else:
            min_max_norm(df, numeric_feature)

    return pd.get_dummies(df, columns=categorical_features)


def plot_confusing_matrix(y_true, y_pred, n_categories, outcome_labels):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(n_categories)))
    plot_confusion_matrix(conf_mat=cm, class_names=outcome_labels, figsize=(10, 10), show_normed=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Target')
    plt.xlabel('Predicted')
    plt.show()


def get_mlp(inp_dim, hidden_dim, out_dim):
    mlp = nn.Sequential(
        nn.Linear(inp_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )
    return mlp


def soft_cross_entropy(student_logit, teacher_logit):
    '''
    :param student_logit: logit of the student arch (without softmax norm)
    :param teacher_logit: logit of the teacher arch (already softmax norm)
    :return: CE loss value.
    '''
    return -(teacher_logit * torch.nn.functional.log_softmax(student_logit, 1)).sum()/student_logit.shape[0]


def kl_divergence(student_logit, teacher_logit):
    return torch.nn.KLDivLoss(reduction='batchmean')(torch.nn.functional.log_softmax(student_logit, 1), teacher_logit)


if __name__ == '__main__':
    import os
    import pandas as pd
    from collections import Counter
    from sklearn.preprocessing import LabelEncoder
    csv_dir = r'F:\入侵检测数据集\Bot-IoT'
    df = pd.DataFrame()
    for csv_path in os.listdir(csv_dir):
        csv_path = csv_dir + '\\' + csv_path
        # sport和dport中存在16进制的数
        df = pd.concat([df, pd.read_csv(csv_path, dtype={'sport': str, 'dport': str})])

    df = data_preprocess_bot_iot(df, is_binary=True)
    # Counter({'DDoS': 1926624, 'DoS': 1650260, 'Reconnaissance': 91082, 'Normal': 477, 'Theft': 79})
    # Counter({0: 1926624, 1: 1650260, 3: 91082, 2: 477, 4: 79})
    df_Y = df.pop('attack').values
    print(Counter(df_Y))
