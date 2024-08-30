import math
import time
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model.model as model

import torch.nn as nn
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

from util import AverageMeter, drop_extra_label, data_preprocess_unsw, get_mlp
from intrusion_detection_datasets import UNSW_NB15Dataset


# 设置随机数种子
seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

total_epochs = 21


class MeanShift(nn.Module):
    def __init__(self, arch, m=0.99, mem_bank_size=128000, topk=5):
        super(MeanShift, self).__init__()

        # save parameters
        self.m = m
        self.mem_bank_size = mem_bank_size
        self.topk = topk

        # create encoders and projection layers
        # both encoders should have same arch

        self.encoder_q = model.__dict__[arch]()
        self.encoder_t = model.__dict__[arch]()

        # save output embedding dimensions
        # feat_dim == 128
        feat_dim = self.encoder_q.fc.in_features
        hidden_dim = feat_dim * 2
        proj_dim = feat_dim // 4

        # projection layers
        self.encoder_t.fc = get_mlp(feat_dim, hidden_dim, proj_dim)
        self.encoder_q.fc = get_mlp(feat_dim, hidden_dim, proj_dim)

        # prediction layer
        self.predict_q = get_mlp(proj_dim, hidden_dim, proj_dim)

        # copy query encoder weights to target encoder
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data.copy_(param_q.data)
            param_t.requires_grad = False

        print("using mem-bank size {}".format(self.mem_bank_size))
        # setup queue (For Storing Random Targets)
        self.register_buffer('queue', torch.randn(self.mem_bank_size, proj_dim))
        # normalize the queue embeddings
        self.queue = nn.functional.normalize(self.queue, dim=1)
        # initialize the labels queue (For Purity measurement)
        self.register_buffer('labels', -1*torch.ones(self.mem_bank_size).long())
        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data = param_t.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, targets, labels):
        batch_size = targets.shape[0]

        ptr = int(self.queue_ptr)
        assert self.mem_bank_size % batch_size == 0 

        # replace the targets at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = targets
        self.labels[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.mem_bank_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_t, labels, epoch, epochs):
        # im_q: img for online encoder
        # img_t: img for target encoder
        # encoder_q: online encoder  encoder_t: target encoder
        feat_q = self.encoder_q(im_q)
        # compute predictions for instance level regression loss
        query = self.predict_q(feat_q)
        query = nn.functional.normalize(query, dim=1)

        # compute target features
        with torch.no_grad():
            # update the target encoder
            self._momentum_update_target_encoder()

            # shuffle targets
            shuffle_ids, reverse_ids = get_shuffle_ids(im_t.shape[0])
            im_t = im_t[shuffle_ids]

            # forward through the target encoder
            current_target = self.encoder_t(im_t)
            current_target = nn.functional.normalize(current_target, dim=1)

            # undo shuffle
            current_target = current_target[reverse_ids].detach()
            self._dequeue_and_enqueue(current_target, labels)

        # calculate mean shift regression loss
        targets = self.queue.clone().detach()
        # calculate distances between vectors
        # current_target shape  == >  (batch_size, proj_dim)
        # targets shape  ==>  (mem_bank_size, proj_dim)
        # query(经过了prediction_stage) shape  ==>  (batch_size, proj_dim)
        dist_t = 2 - 2 * torch.einsum('bc,kc->bk', [current_target, targets])
        dist_q = 2 - 2 * torch.einsum('bc,kc->bk', [query, targets])

        _, nn_near_index = dist_t.topk(self.topk, dim=1, largest=False)
        _, nn_far_index = dist_t.topk(self.topk, dim=1)
        nn_near_dist_q = torch.gather(dist_q, 1, nn_near_index)
        nn_far_dist_q = torch.gather(dist_q, 1, nn_far_index)

        if epoch < epochs / 2:
            loss = (nn_near_dist_q.sum(dim=1) / self.topk).mean() - (epochs-epoch)/epochs * (nn_far_dist_q.sum(dim=1) / self.topk).mean()
        else:
            loss = (nn_near_dist_q.sum(dim=1) / self.topk).mean()

        return loss


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


class TwoCropsTransform:
    def __init__(self, t_t, q_t):
        self.q_t = q_t
        self.t_t = t_t
        print('======= Query transform =======')
        print(self.q_t)
        print('===============================')
        print('======= Target transform ======')
        print(self.t_t)
        print('===============================')

    def __call__(self, x):
        q = self.q_t(x)
        t = self.t_t(x)
        return [q, t]


class RandomShuffle(object):
    def __init__(self, p=0.9):
        self.p = p

    def __call__(self, img):
        if np.random.uniform(0, 1) < self.p:
            img = img.reshape(img.size(0), -1)
            np.random.shuffle(img)
            return img.reshape(img.size(0), 14, 14)
        else:
            return img

    def __str__(self):
        return "RandomShuffle(p={})".format(self.p)


# Create train loader
def get_train_loader(train_data, test_data, batch_size):

    aug_strong = transforms.Compose([
        # RandomShuffle(),
        transforms.RandomCrop(size=(12, 12)),
        transforms.Resize(size=(14, 14)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5)
    ])

    aug_weak = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
    ])

    # TwoCropsTransform   ==> 对img进行两个data aug, 并返回[aug_img_online, aug_img_target]
    train_dataset = UNSW_NB15Dataset(train_data, test_data, transform=TwoCropsTransform(t_t=aug_strong, q_t=aug_weak))

    print('==> train dataset')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, drop_last=True)

    return train_loader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_csv_path = r'train_csv_path.csv'
    test_csv_path = r'test_csv_path.csv'
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)
    df_X = drop_extra_label(df_train, df_test, ['id', 'label'])
    df_Y = LabelEncoder().fit_transform(df_X.pop('attack_cat').values)
    df_X = data_preprocess_unsw(df_X).values.astype(np.float32)
    train_loader = get_train_loader(df_X, df_Y, batch_size=1024)
    mean_shift = MeanShift('cnn', m=0.999, mem_bank_size=10240, topk=5).to(device)

    params = [p for p in mean_shift.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    # cudnn.benchmark = True
    loss_list = []
    # routine
    for epoch in range(1, total_epochs):

        print("==> training...")

        time1 = time.time()
        train(epoch, train_loader, mean_shift, optimizer, loss_list, device)

        time2 = time.time()
        print('\nepoch {}, total time {:.2f}s'.format(epoch, time2 - time1))

    torch.save(mean_shift.state_dict(), 'representation_msf.pt')
    plt.plot(loss_list, color='blue')
    plt.show()


def train(epoch, train_loader, mean_shift, optimizer, loss_list, device):
    mean_shift.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    trained_samples = 0
    # train_dataset中TwoCropsTransform对img进行两次data aug, 并返回[aug_img_online, ang_img_target]
    for idx, (indices, (im_q, im_t), labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        im_q = im_q.to(device)
        im_t = im_t.to(device)
        labels = labels.to(device)

        # ===================forward=====================
        loss = mean_shift(im_q=im_q, im_t=im_t, labels=labels, epoch=epoch, epochs=total_epochs)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trained_samples += len(im_q)
        progress = math.ceil(idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')

        # ===================meters=====================
        loss_meter.update(loss.item(), im_q.size(0))

        # torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        loss_list.append(loss_meter.val)

    return loss_meter.avg


if __name__ == '__main__':
    main()
