import torch
from torchviz import make_dot

# 指定.pt文件的路径
file_path = "outputs/0.0001/Shield_Mcts/Section16/20230921-192004/models/checkpoint.pt"
file_path2 = "outputs/0.0001/Shield_Mcts/Section16/20230921-191654/models/checkpoint.pt"
# 加载.pt文件
model = torch.load(file_path)
model2 = torch.load(file_path2)
