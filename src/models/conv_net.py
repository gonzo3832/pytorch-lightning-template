"""二次元データ（画像用）モデル
必ずoutputはsqueeze()して次元削減すること！

"""
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn.functional as F


class NN_Linear(nn.Module):
    def __init__(self, in_sz=841, out_sz=1, layers=[120, 84]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_sz)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)
        x = Swish()(self.fc1(x))
        x = Swish()(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1)


class MyConv1(nn.Module):
    def __init__(self, in_c=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 16, 3, 1, 1)
        self.norm = BatchNorm2d(self.conv1.out_channels)
        self.GAP = GlobalAvgPool2d()
        self.fc = nn.Linear(self.conv1.out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = Swish()(x)
        x = self.GAP(x)
        x = self.fc(x)

        return x.squeeze(1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]) .view(-1, x.size(1))


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]) .view(-1, x.size(1))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x*torch.sigmoid(x)


class myConv2(nn.Module):
    def __init__(self, in_c=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 20, 3, 1, 1)
        self.norm1 = BatchNorm2d(self.conv1.out_channels)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(self.conv1.out_channels, 20, 3, 1, 1)
        self.norm2 = BatchNorm2d(self.conv2.out_channels)

        self.GAP = GlobalAvgPool2d()
        self.fc = nn.Linear(self.conv2.out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = Swish()(self.norm1(x))
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = Swish()(self.norm2(x))

        x = self.GAP(x)
        x = self.fc(x)
        return(x).squeeze(1)


class Res10(nn.Module):
    def __init__(self, in_c=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 48, 3, 1, 1)
        self.bn1 = BatchNorm2d(self.conv1.out_channels)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.bblock1 = BBlock(self.conv1.out_channels)
        self.bblock2 = BBlock(self.bblock1.in_c)

        self.conv2 = nn.Conv2d(self.bblock2.in_c, 96, 1, 1)

        self.bblock3 = BBlock(self.conv2.out_channels)
        self.bblock4 = BBlock(self.bblock3.in_c)

        self.GAP = GlobalAvgPool2d()
        self.fc = nn.Linear(self.bblock4.in_c, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = Swish()(x)
        x = self.pool1(x)

        x = self.bblock1(x)
        x = self.bblock2(x)

        x = self.conv2(x)

        x = self.bblock3(x)
        x = self.bblock4(x)

        x = self.GAP(x)
        x = self.fc(x)

        return x.squeeze(1)


class Res18(nn.Module):
    def __init__(self, in_c=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 64, 3, 1, 1)
        self.bn1 = BatchNorm2d(self.conv1.out_channels)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.bblock1 = BBlock(self.conv1.out_channels)
        self.bblock2 = BBlock(self.bblock1.in_c)

        self.conv2 = nn.Conv2d(self.bblock2.in_c, 128, 1, 1)

        self.bblock3 = BBlock(self.conv2.out_channels)
        self.bblock4 = BBlock(self.bblock3.in_c)

        self.conv3 = nn.Conv2d(self.bblock4.in_c, 256, 1, 1)

        self.bblock5 = BBlock(self.conv3.out_channels)
        self.bblock6 = BBlock(self.bblock5.in_c)

        self.conv4 = nn.Conv2d(self.bblock6.in_c, 512, 1, 1)

        self.bblock7 = BBlock(self.conv4.out_channels)
        self.bblock8 = BBlock(self.bblock7.in_c)

        self.GAP = GlobalAvgPool2d()
        self.fc = nn.Linear(self.bblock8.in_c, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = Swish()(x)
        x = self.pool1(x)

        x = self.bblock1(x)
        x = self.bblock2(x)

        x = self.conv2(x)

        x = self.bblock3(x)
        x = self.bblock4(x)

        x = self.conv3(x)

        x = self.bblock5(x)
        x = self.bblock6(x)

        x = self.conv4(x)

        x = self.bblock7(x)
        x = self.bblock8(x)

        x = self.GAP(x)
        x = self.fc(x)

        return x.squeeze(1)


class BBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.in_c = in_c
        bottle_c = in_c // 4

        self.conv1 = nn.Conv2d(in_c, bottle_c, 1)
        self.bn1 = BatchNorm2d(bottle_c)

        self.conv2 = nn.Conv2d(bottle_c, bottle_c, 3, 1, 1)
        self.bn2 = BatchNorm2d(bottle_c)

        self.conv3 = nn.Conv2d(bottle_c, in_c, 1)
        self.bn3 = BatchNorm2d(in_c)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = Swish()(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = Swish()(h)
        h = self.conv3(h)
        h = self.bn3(h)
        x = Swish()(x + h)

        return x
