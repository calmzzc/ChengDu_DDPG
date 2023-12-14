import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


def label_font():
    return FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=15)  # 系统字体路径，此处是mac的


# pd.read_excel('test_data.xlsx', sheet_name='Sheet1', usecols='A,DA', nrows=11, header=2, index_col=0, engine='openpyxl')
def smooth(data, sm=2):
    smooth_data = []
    if sm > 1:
        for d in data:
            y = np.ones(sm) * 1.0 / sm
            d = np.convolve(y, d, "same")

            smooth_data.append(d)
    return smooth_data


def get_data():
    df = pd.read_excel('processed_rewards/ma_rewards_sec4.xlsx')
    a = df.values[:, 0]
    b = df.values[:, 1]
    c = df.values[:, 2]
    d = df.values[:, 3]
    e = df.values[:, 4]
    f = df.values[:, 5]
    g = df.values[:, 6]
    h = df.values[:, 7]
    i = df.values[:, 8]
    j = df.values[:, 9]
    k = df.values[:, 10]
    l = df.values[:, 11]
    m = df.values[:, 12]
    n = df.values[:, 13]
    o = df.values[:, 14]
    p = df.values[:, 15]
    q = df.values[:, 16]
    r = df.values[:, 17]
    # j = df.values[:, 9]
    # print(a)
    # print(b)
    # print(c)
    test1 = np.array([a, b, c])
    # test1 = smooth(test1, 3)
    test2 = np.array([d, e, f])
    # test2 = smooth(test2, 3)
    test3 = np.array([g, h, i])
    test4 = np.array([j, k, l])
    test5 = np.array([m, n, o])
    test6 = np.array([p, q, r])
    return test1, test2, test3, test4, test5, test6


#
# def get_data2():
#     df3 = pd.read_excel('unsafe_data_o6.xlsx')
#     a1 = df3.values[:, 0]
#     b1 = df3.values[:, 1]
#     c1 = df3.values[:, 2]
#     d1 = df3.values[:, 3]
#     e1 = df3.values[:, 4]
#     f1 = df3.values[:, 5]
#     g1 = df3.values[:, 6]
#     h1 = df3.values[:, 7]
#     i1 = df3.values[:, 8]
#     j1 = df3.values[:, 9]
#     # print(a)
#     # print(b)
#     # print(c)
#     test1 = np.array([a1, b1, c1, d1, e1])
#     # test1 = smooth(test1, 5)
#     test2 = np.array([f1, g1, h1, i1, j1])
#     # test2 = smooth(test2, 5)
#     return test1, test2


data = get_data()
# font = {'family': 'Times New Roman'}
label = ['SSA-SAC', 'Shield-SAC', 'SAC', 'SSA-DDPG', 'Shield-DDPG', 'DDPG']
df2 = []
for i in range(len(data)):
    df2.append(pd.DataFrame(data[i]).melt(var_name='Episode', value_name='Reward'))
    df2[i]['Algo'] = label[i]

df2 = pd.concat(df2)  # 合并
df2.index = range(len(df2))
# print(df2)
fig = plt.figure(dpi=150)
fig.subplots_adjust(left=0.16, right=0.94, top=0.94, bottom=0.1)  # 设置子图与四个边的间距
ax=sns.lineplot(x="Episode", y="Reward", hue="Algo", style="Algo", data=df2, legend=False)
# ax.set_yticklabels(['{:,.2e}'.format(tick) for tick in ax.get_yticks()])
# ax.set_xticklabels(ax.get_xticks(), fontfamily='Times New Roman')
# ax.set_yticklabels(ax.get_yticks(), fontfamily='Times New Roman')
plt.legend(labels=['SSA-SAC', 'Shield-SAC', 'SAC', 'SSA-DDPG', 'Shield-DDPG', 'DDPG'], loc='lower right')
# plt.title("some loss")
plt.show()

# data2 = get_data2()
# label2 = ['DDPG', 'Shield-DDPG']
# df4 = []
# for i in range(len(data2)):
#     df4.append(pd.DataFrame(data2[i]).melt(var_name='Episode', value_name='Protection or Punishment Times'))
#     df4[i]['Algo'] = label2[i]
#
# df4 = pd.concat(df4)  # 合并
# df4.index = range(len(df4))
# # print(df2)
# plt.figure(dpi=150)
# sns.lineplot(x="Episode", y="Protection or Punishment Times", hue="Algo", style="Algo", data=df4, legend=False)
# plt.legend(labels=['DDPG', 'Shield-DDPG'], loc='upper right')
# # plt.title("some loss")
# plt.show()
