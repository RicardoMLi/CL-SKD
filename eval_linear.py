import torch
import random
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

import model.model as model
from intrusion_detection_datasets import UNSW_NB15Dataset
from util import drop_extra_label, data_preprocess_unsw, plot_confusing_matrix

# 设置随机数种子
seed = 321
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

epochs = 70
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_csv_path = r'train_csv_path.csv'
test_csv_path = r'test_csv_path.csv'

df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)
df_X = drop_extra_label(df_train, df_test, ['id', 'attack_cat'])
df_Y = df_X.pop('label').values
df_X = data_preprocess_unsw(df_X).values.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.25, random_state=666)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=666)


train_ds = UNSW_NB15Dataset(x_train, y_train)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

val_ds = UNSW_NB15Dataset(x_val, y_val)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=True)

test_ds = UNSW_NB15Dataset(x_test, y_test)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=True)

target_encoder = model.__dict__['cnn']().to(device)
target_encoder.fc = nn.Sequential()
ckpt = {k.replace('encoder_q.', ''): v for k, v in torch.load('representation_k_100.pt').items()}
state_dict = {}

for m_key, m_val in target_encoder.state_dict().items():
    if m_key in ckpt:
        state_dict[m_key] = ckpt[m_key]
    else:
        state_dict[m_key] = m_val
        print('not copied => ' + m_key)

target_encoder.load_state_dict(state_dict)
print(target_encoder)

c_net = model.__dict__['classifier']().to(device)
optimizer = torch.optim.AdamW(c_net.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for param in target_encoder.parameters():
    param.requires_grad = False


def train(net, classifer, device, train_loader, optimizer, epoch, criterion):
    classifer.train()
    trained_samples = 0
    for batch_idx, (_, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = classifer(net(data))
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')


def test(net, classifer, loader, criterion, show_matrix=False):
    classifer.eval()
    net.eval()
    correct = 0
    test_loss = 0.0
    total_target = torch.zeros(len(loader.dataset), dtype=torch.int64).to(device)
    total_pred = torch.zeros(len(loader.dataset), dtype=torch.int64).to(device)
    with torch.no_grad():
        for batch_idx, (input_indices, inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            total_target[input_indices] = targets
            # model output
            outputs = classifer(net(inputs))
            test_loss += criterion(outputs, targets.long()).item()
            predicted = outputs.argmax(dim=1)
            total_pred[input_indices] = predicted
            correct += predicted.eq(targets).sum().item()

    pre_score = precision_score(total_target.cpu().numpy(), total_pred.cpu().numpy(), average='weighted')
    re_score = recall_score(total_target.cpu().numpy(), total_pred.cpu().numpy(), average='weighted')
    f_score = f1_score(total_target.cpu().numpy(), total_pred.cpu().numpy(), average='weighted')

    if show_matrix:
        outcome_labels = ['Normal', 'Attack']
        plot_confusing_matrix(total_target.cpu().numpy(), total_pred.cpu().numpy(), 2, outcome_labels)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        test_loss/len(loader.dataset), correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

    return correct / len(loader.dataset), pre_score, re_score, f_score


best_acc = 0
val_list = []
for epoch in range(1, epochs+1):
    train(target_encoder, c_net, device, train_loader, optimizer, epoch, criterion)
    acc, pre, recall, f1 = test(target_encoder, c_net, val_loader, criterion)
    print(f'Accuracy: {acc}, Precision: {pre}, Recall: {recall}, F1 score: {f1}')
    val_list.append(acc)

    if best_acc < acc:
        best_acc = acc
        torch.save(c_net.state_dict(), 'best_classifier.pt')
        torch.save(target_encoder.state_dict(), 'best_target_encoder.pt')


best_classifier = model.__dict__['classifier']().to(device)
best_target_encoder = model.__dict__['cnn']().to(device)
best_target_encoder.fc = nn.Sequential()

best_classifier.load_state_dict(torch.load('best_classifier.pt'))
best_target_encoder.load_state_dict(torch.load('best_target_encoder.pt'))
acc, pre, recall, f1 = test(best_target_encoder, best_classifier, test_loader, criterion, show_matrix=True)
print(f'Accuracy: {acc}, Precision: {pre}, Recall: {recall}, F1 score: {f1}')

plt.plot(val_list, color='red')
plt.show()
