import torch
import torch.nn as nn
import torch.nn.functional as F

class IDDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(IDDQN, self).__init__()
        # 输入层 -> 3个隐藏层 -> 输出层 [cite: 308]
        # 隐藏层神经元数量为 128 [cite: 448]
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)
        
    def forward(self, x):
        # 激活函数使用 ReLU [cite: 331]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)