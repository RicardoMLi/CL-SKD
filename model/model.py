import torch
import math
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['botnet', 'classifier', 'cnn', 'student']


class MHSA(nn.Module):
    def __init__(self, n_dims, width=4, height=4, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class BoTNetBottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(BoTNetBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BoTNet(nn.Module):
    def __init__(self, block, num_blocks, filter_num, resolution=(14, 14), heads=4):
        super(BoTNet, self).__init__()
        self.in_planes = filter_num[0]
        self.resolution = list(resolution)

        self.conv1 = nn.Conv2d(1, filter_num[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_num[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, filter_num[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, filter_num[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, filter_num[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, filter_num[3], num_blocks[3], stride=2, heads=heads, mhsa=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filter_num[-1]*2, filter_num[-2])

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution))
            if stride == 2:
                self.resolution[0] = math.ceil(self.resolution[0]/2)
                self.resolution[1] = math.ceil(self.resolution[1]/2)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, fc_dim=128, in_channel=1, width=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(self.base * 8 * block.expansion, fc_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, layer=7):
        if layer <= 0:
            return x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if layer == 1:
            return x
        x = self.layer1(x)
        if layer == 2:
            return x
        x = self.layer2(x)
        if layer == 3:
            return x
        x = self.layer3(x)
        if layer == 4:
            return x
        x = self.layer4(x)
        if layer == 5:
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if layer == 6:
            return x
        x = self.fc(x)

        return x


class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Depthwise_Separable_Conv, self).__init__()

        self.depth_wise = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, groups=in_channel, bias=False)
        self.point_wise = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)

        return x


class Student(nn.Module):
    def __init__(self):

        super(Student, self).__init__()
        self.conv1 = Depthwise_Separable_Conv(in_channel=1, out_channel=32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = Depthwise_Separable_Conv(in_channel=32, out_channel=64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = Depthwise_Separable_Conv(in_channel=64, out_channel=128, kernel_size=6)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CNN(nn.Module):
    def __init__(self):

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=6)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class DNN(nn.Module):
    def __init__(self):

        super(DNN, self).__init__()
        self.conv1 = nn.Linear(196, 32)
        # self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Linear(32, 64)
        # self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Linear(64, 128)
        # self.bn3 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class RNN(nn.Module):
    def __init__(self):

        super(RNN, self).__init__()
        self.conv1 = nn.LSTM(196, 128, batch_first=True)

        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        output, (_, _) = self.conv1(x)
        x = torch.squeeze(output, dim=1)
        x = self.fc(x)

        return x


class Classfication(nn.Module):
    def __init__(self, channel_in, num_classes=2):
        super(Classfication, self).__init__()
        self.bn = nn.BatchNorm1d(channel_in)
        self.fc = nn.Linear(channel_in, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)

        return x


def botnet(filter_num=[32, 64, 128, 256]):
    return BoTNet(BoTNetBottleneck, [2, 2, 2, 2], filter_num)


def classifier():
    return Classfication(128, num_classes=10)


def cnn():
    return CNN()


def student():
    return Student()


if __name__ == '__main__':
    from thop import profile
    model = Student()
    x = torch.randn(size=(10, 1, 14, 14))

    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


