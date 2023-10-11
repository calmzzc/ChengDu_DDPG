import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建数据集
df = pd.read_excel('test.xlsx')
tensor_x = []
tensor_y = []
for row in df.itertuples(index=False):
    tensor = torch.tensor([row.col1, row.col2]).to(device)
    tensor2 = torch.tensor([row.col3]).to(device)
    tensor_x.append(tensor)
    tensor_y.append(tensor2)


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    # 实例化神经网络模型和损失函数


model = Net().to(device)
criterion = nn.MSELoss()

# 选择优化算法和训练数据集
optimizer = optim.Adam(model.parameters())

# 训练神经网络模型
for epoch in range(10000):
    for i in range(len(tensor_x)):
        action = model(tensor_x[i])
        loss = criterion(action, tensor_y[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{100}], Loss: {loss.item()}")

# 使用训练好的模型进行预测
with torch.no_grad():
    y_pred = model(tensor_x)
    print(y_pred)
