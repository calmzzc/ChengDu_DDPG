# import numpy as np
# import matplotlib.pyplot as plt
#
# # 设定高斯分布的均值和标准差
# mean = 0
# std_dev = 1
# np.random.seed(1)
# # 使用numpy的random.normal函数生成100个样本
# samples = np.random.normal(mean, std_dev, 100)
#
# # 使用matplotlib来可视化生成的样本
# plt.hist(samples, bins=20, density=True, alpha=0.5, color='g')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Gaussian Distribution')
# plt.show()
# from scipy.interpolate import interp1d
# import numpy as np
#
# # 已知数据点
# x = np.array([0, 5])
# y = np.array([0, -1])
#
# # 创建插值函数
# f = interp1d(x, y)
#
# # 使用插值函数计算新数据点
# xnew = np.linspace(0, 5, num=6, endpoint=True)
# ynew = f(xnew)  # use interpolation function returned by `interp1d`
#
# # 可视化结果
# import matplotlib.pyplot as plt
#
# plt.plot(x, y, 'o', xnew, ynew, '-')
# plt.show()
# import random
#
# buffer = []
# buffer.append([1, 2, 3, 4, 5])
# buffer.append([6, 7, 8, 9, 0])
# buffer.append([2, 3, 4, 5, 6])
# buffer.append([7, 8, 9, 0, 1])
# sample_batch = random.sample(buffer, 2)
# print(sample_batch)
import torch
import torch.nn as nn
import torch.optim as optim


# 定义模型结构
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_variable, output_variable):
#         super(NeuralNetwork, self).__init__()
#         self.layer1 = nn.Linear(input_variable.shape[1], 64)
#         self.layer2 = nn.Linear(64, 32)
#         self.layer3 = nn.Linear(32, output_variable.shape[1])
#         self.init_w=3e-3
#         self.layer3.weight.data.uniform_(-self.init_w, self.init_w)
#         self.layer3.bias.data.uniform_(-self.init_w, self.init_w)
#
#     def forward(self, x):
#         x = torch.relu(self.layer1(x))
#         x = torch.relu(self.layer2(x))
#         x = torch.tanh(self.layer3(x))
#         return x
#
#
# # 假设我们有一些输入和输出数据
# input_variable = torch.rand((10, 5))  # 假设有100个样本，每个样本有5个特征
# output_variable = torch.rand((10, 3))  # 假设有100个样本，每个样本期望有3个输出
#
# # 初始化模型和优化器
# model = NeuralNetwork(input_variable, output_variable)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 定义损失函数
# criterion = nn.MSELoss()
#
# # 训练模型
# for epoch in range(5000):  # 迭代100轮
#     # 前向传播
#     outputs = model(input_variable)
#     print(outputs)
#     loss = criterion(outputs, output_variable)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     # 打印损失信息（可选）
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch {epoch + 1}/{100}, Loss: {loss.item()}')

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    # 实例化模型


model = MyModel(input_size=784, hidden_size=128, output_size=10)

# 构造输入数据
input_data = torch.randn(32, 784)  # 假设有32个样本，每个样本的特征维度为784

# 前向传播
output = model(input_data)
print(output.shape)  # 输出结果的形状应为[32, 10]
