import numpy as np
import matplotlib.pyplot as plt

# 设定高斯分布的均值和标准差
mean = 0
std_dev = 1
np.random.seed(1)
# 使用numpy的random.normal函数生成100个样本
samples = np.random.normal(mean, std_dev, 100)

# 使用matplotlib来可视化生成的样本
plt.hist(samples, bins=20, density=True, alpha=0.5, color='g')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Gaussian Distribution')
plt.show()
