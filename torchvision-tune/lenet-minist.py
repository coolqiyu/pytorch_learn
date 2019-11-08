"""
自己实现的LeNet，并在MINIST上测试
--2019-11-08
用原来的数据集进行测试，效果不错
用自己手写的几张图片测试结果为0：
 - 没有对数据进行处理，测试集与训练集的数据分布不一致
 - 进行数据增强，增加对比度
"""

import os
import matplotlib.pyplot as plt  # 用于显示图像
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, 5, stride=1, padding=2)
        self.fc1 = nn.Linear(120 * 7 * 7, 84)
        self.fc2 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.flattern(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


    def flattern(self, x):
        """
        展开成一维的数组
        :param x:
        :return:
        """
        size = x.size()[1:] # 除了batch_size
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def get_data_loader():
    # 训练数据集
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=64, shuffle=True, num_workers=4)
    # 测试数据集
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=64, shuffle=True, num_workers=4)
    return train_loader, test_loader


def train_model(net, data_loader, epoch, criterion, optimizer):
    net.train() # 设置为训练模式
    for epoch_i in range(epoch):
        running_loss = 0
        for data_i, data in enumerate(data_loader, 0):
            # 每次获取一个batch
            inputs, labels = data
            # 把梯度置零，也就是把loss关于weight的导数变成0.
            optimizer.zero_grad()

            # 前向传播
            outputs = net(inputs)
            # 计算loss
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 执行一次优化，很重要
            optimizer.step()

            running_loss += loss.item()
            # print(data_i)
            if data_i % 300 == 299:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch_i + 1, data_i + 1, running_loss / 300))
                running_loss = 0.0
    torch.save(net.state_dict(), "./lenet.pth")
    print("Finish Train")


def test_model(net, test_loader):
    total = 0
    correct = 0
    with torch.no_grad():
        for data_i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            labels = torch.tensor([x for x in labels])
            total += labels.size(0)
            print(predicted)
            print(labels)
            correct += (predicted == labels).sum().item()
    print("accuracy: {}".format(correct/total))


class MyMINISTDataset(Dataset):
    def __init__(self, root):
        self.root = root
        # 要先给出数据的均值和方差，然后再这里做归一化
        # # ToTensor会把输入的dst当成图片[长，宽，通道]，在转换过程中会改变顺序
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2842,), (0.0710,))])
        self.imgs = list(os.listdir(os.path.join(self.root, "MYMINIST")))


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "MYMINIST", self.imgs[idx])
        label = int(os.path.split(self.imgs[idx])[1][:-4])
        img = io.imread(img_path, as_gray=True)
        dst = transform.resize(img, (28, 28)).astype(np.float32)
        # 增强对比度和数据取值的反向，以平均值作为分界线
        mean = np.resize(dst, 28 * 28).mean() - 0.05
        for idx, d in enumerate(dst):
            for idy, dy in enumerate(d):
                if dy >= mean:
                    dst[idx][idy] = 0 if 1 - dy - 0.2 < 0 else 1 - dy - 0.2
                else:
                    dst[idx][idy] = 1 if 1 - dy + 0.2 > 1 else 1 - dy + 0.2
        io.imsave("../tmp/n{}.jpg".format(label), dst)
        dst.resize(28, 28, 1)
        dst = self.transform(dst)
        return dst, label


def show_img(data_loader):
    for data_i, data in enumerate(data_loader, 0):
        imgs, labels = data
        for img in imgs:
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.show()


def get_mean_std():
    """
    计算自己手写的几个字的均值和方差
    :return:
    """
    mean = 0.0
    std = 0.0
    my_minist = MyMINISTDataset(os.getcwd())
    data_loader = torch.utils.data.DataLoader(my_minist, batch_size=1)
    for data_i, data in enumerate(data_loader, 0):
        images, labels = data
        for img in images:
            img_shape = 1
            for s in img.shape:
                img_shape *= s
            img = np.resize(img, img_shape)
            mean += img.mean()
            std += img.std()

    mean = mean/4
    std = mean/4
    return mean, std


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    # # 获取数据
    train_loader, test_loader = get_data_loader()
    # # 显示一下获取到的数据
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))

    # # 训练网络并保存
    # net = LeNet()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # train_model(net, train_loader, 10, criterion, optimizer)
    #
    # # 加载网络并测试
    # load_net = LeNet()
    # load_net.load_state_dict(torch.load("../data/lenet.pth"))
    # load_net.eval()
    # test_model(load_net, test_loader)

    # 用自己的数据集进行测试
    my_minist = MyMINISTDataset(os.getcwd() + "\\..\\data")
    data_loader = torch.utils.data.DataLoader(my_minist, batch_size=1)
    dataiter_2 = iter(data_loader)
    images_2, labels_2 = dataiter_2.next()
    load_net = LeNet()
    load_net.load_state_dict(torch.load("../data/lenet.pth"))
    load_net.eval()
    test_model(load_net, data_loader)

    # 计算mean和方差
    # mean, std = get_mean_std()
    # print(mean)
    # print(std)

