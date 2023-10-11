import matplotlib.pyplot as plt
import numpy as np

# 定义数据
data1 = [1, 2, 3, 4, 5, 3, 4, 5]
data2 = [2, 3, 4, 5, 6, 3, 4, 5]
data3 = [5, 6, 7, 8, 9, 3, 4, 5]

# 定义x轴标签
x_labels = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
x_labels_subplot = [x_labels[i:i + 3] for i in range(0, len(x_labels), 3)]  # convert to list of lists with 3 elements each

# 使用bar函数绘制柱状图，将每个x_label对应的三组data画成三个柱
fig, axs = plt.subplots(len(x_labels_subplot), 1, figsize=(10, 30))  # create figure with multiple axes

for i, ax in enumerate(axs):
    ax.bar(x_labels_subplot[i], data1[i:i + 3], color='blue')  # plot the first dataset
    ax.bar(x_labels_subplot[i], data2[i:i + 3], color='green')  # plot the second dataset
    ax.bar(x_labels_subplot[i], data3[i:i + 3], color='red')  # plot the third dataset

    ax.set_xticks(range(len(x_labels_subplot[i])))  # set x ticks according to the subplot labels
    ax.set_xticklabels(x_labels_subplot[i])  # set x tick labels according to the subplot labels

plt.ylabel('Y-axis label')
plt.tight_layout()  # to avoid overlap between subplots
plt.show()