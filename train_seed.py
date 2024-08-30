import math
import time
import torch.optim
import torch.nn.parallel
import numpy as np
import pandas as pd
import torch.nn as nn
import model.model as model
import matplotlib.pyplot as plt
import torch.utils.data.distributed

from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

from intrusion_detection_datasets import UNSW_NB15Dataset
from util import drop_extra_label, data_preprocess_unsw, soft_cross_entropy, get_mlp, kl_divergence


class SEED(nn.Module):
    """
    Build a SEED model for Self-supervised Distillation: a student encoder, a teacher encoder (stay frozen),
    and an instance queue.
    Adapted from MoCo, He, Kaiming, et al. "Momentum contrast for unsupervised visual representation learning."
    """
    def __init__(self, student_arch, teacher_arch, K=65536, t=0.07, temp=1e-4, k=100):
        """
        dim:        feature dimension (default: 128)
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        k:          select top k elements to save computational efforts
        """
        super(SEED, self).__init__()

        self.K = K
        self.t = t
        self.temp = temp
        self.k = k
        self.student = model.__dict__[student_arch]()
        self.teacher = model.__dict__[teacher_arch]()

        # create the Teacher/Student encoders
        feat_dim = self.teacher.fc.in_features
        hidden_dim = feat_dim * 2
        proj_dim = feat_dim // 4

        self.student.fc = get_mlp(feat_dim, hidden_dim, proj_dim)
        self.teacher.fc = get_mlp(feat_dim, hidden_dim, proj_dim)
        self.predict_q = get_mlp(proj_dim, hidden_dim, proj_dim)

        # not update by gradient
        for param_k in self.teacher.parameters():
            param_k.requires_grad = False

        for param_k in self.predict_q.parameters():
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(proj_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # queue updation
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity as in MoCo-v2

        # replace the keys at ptr (de-queue and en-queue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        # move pointer
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, image):
        """
        Input:
            image: a batch of images
        Output:
            student logits, teacher logits
        """
        # compute query features
        s_emb = self.student(image)  # NxC
        s_emb = nn.functional.normalize(s_emb, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            t_emb = self.predict_q(self.teacher(image))  # keys: NxC
            t_emb = nn.functional.normalize(t_emb, dim=1)

        # cross-Entropy Loss
        # logit_stu shape: [batch_size, men_size]
        logit_stu = torch.einsum('nc,ck->nk', [s_emb, self.queue.clone().detach()])
        logit_tea = torch.einsum('nc,ck->nk', [t_emb, self.queue.clone().detach()])
        # 取出前100个最大的即最相似再做cross entropy，节省计算开销
        logit_stu, _ = logit_stu.topk(self.k, dim=1)
        logit_tea, _ = logit_tea.topk(self.k, dim=1)

        # logit_s_p shape: [batch_size, 1]
        logit_s_p = torch.einsum('nc,nc->n', [s_emb, t_emb]).unsqueeze(-1)
        logit_t_p = torch.einsum('nc,nc->n', [t_emb, t_emb]).unsqueeze(-1)

        # logit_stu shape: [batch_size, men_size]
        logit_stu = torch.cat([logit_s_p, logit_stu], dim=1)
        logit_tea = torch.cat([logit_t_p, logit_tea], dim=1)

        # compute soft labels
        logit_stu /= self.t
        logit_tea = nn.functional.softmax(logit_tea/self.temp, dim=1)

        # de-queue and en-queue
        self._dequeue_and_enqueue(t_emb)

        # loss = -(logit_tea * torch.nn.functional.log_softmax(logit_stu, 1)).sum()/logit_stu.shape[0]
        return logit_stu, logit_tea


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize model object, feed student and teacher into encoders.
    seed = SEED('student', 'cnn', k=100).to(device)
    params = [p for p in seed.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    # load the SSL pre-trained teacher encoder into model.teacher
    ckpt = {}
    for k, v in torch.load('representation_msf.pt').items():
        if k.startswith('encoder_q'):
            ckpt[k.replace('encoder_q', 'teacher')] = v

        if k.startswith('predict_q'):
            ckpt[k] = v

    state_dict = {}

    for m_key, m_val in seed.state_dict().items():
        if m_key in ckpt:
            state_dict[m_key] = ckpt[m_key]
        else:
            state_dict[m_key] = m_val
            print('not copied => ' + m_key)

    seed.load_state_dict(state_dict)
    # clear unnecessary weights
    torch.cuda.empty_cache()

    train_csv_path = r'train_csv_path.csv'
    test_csv_path = r'test_csv_path.csv'
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)
    df_X = drop_extra_label(df_train, df_test, ['id', 'label'])
    df_Y = LabelEncoder().fit_transform(df_X.pop('attack_cat').values)
    df_X = data_preprocess_unsw(df_X).values.astype(np.float32)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
    ])

    train_dataset = UNSW_NB15Dataset(df_X, df_Y, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=True, drop_last=True)

    loss_list = []
    time_list = []
    for epoch in range(1, 21):
        # train for one epoch
        print("==> training...")

        time1 = time.time()
        train(device, train_loader, seed, soft_cross_entropy, optimizer, epoch, loss_list)
        time2 = time.time()
        time_list.append(time2 - time1)
        print('\nepoch {}, total time {:.2f}s'.format(epoch, time2 - time1))

    torch.save(seed.state_dict(), 'representation_k_100.pt')
    plt.plot(loss_list, color='blue')
    plt.show()
    # total cost time: 626.4747228622437s, average time per epoch: 31.323736143112182s
    # total cost time: 615.229086637497, average time per epoch: 30.761454331874848
    print("total cost time: {0}, average time per epoch: {1}".format(sum(time_list), sum(time_list)/len(time_list)))


def train(device, train_loader, model, criterion, optimizer, epoch, loss_list):
    model.train()
    model.teacher.eval()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    trained_samples = 0
    for idx, (_, images, _) in enumerate(train_loader):
        images = images.to(device)
        # compute output
        with torch.cuda.amp.autocast(enabled=True):
            logit, label = model(images)
            loss = criterion(logit, label)
            loss_list.append(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        trained_samples += len(images)
        progress = math.ceil(idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')


if __name__ == '__main__':
    main()
