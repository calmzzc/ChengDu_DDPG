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
import random

buffer = []
buffer.append([1, 2, 3, 4, 5])
buffer.append([6, 7, 8, 9, 0])
buffer.append([2, 3, 4, 5, 6])
buffer.append([7, 8, 9, 0, 1])
sample_batch = random.sample(buffer, 2)
print(sample_batch)
