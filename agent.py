import torch
import torch.optim as optim
import numpy as np
import random
from model import IDDQN

class IDDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 定义 Current Network 和 Target Network [cite: 151]
        self.policy_net = IDDQN(state_dim, action_dim)
        self.target_net = IDDQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 初始化参数相同
        self.target_net.eval() # 目标网络不进行训练
        
        # 优化器 Adam, 学习率 0.0025 [cite: 335, 448]
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0025)
        self.loss_func = torch.nn.MSELoss()
        
        # 经验回放池
        self.memory = []
        self.memory_capacity = 10000# [cite: 448]
        self.batch_size = 64
        self.gamma = 0.9 # 折扣因子 [cite: 448]
        
        # 自适应 Epsilon 参数 [cite: 351]
        self.epsilon_start = 0.9
        self.epsilon_final = 0.01
        self.epsilon_decay = 500
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state):
        self.steps_done += 1
        # 自适应 Epsilon 公式 (Eq. 15)
        eps_threshold = self.epsilon_final + (self.epsilon_start - self.epsilon_final) / \
                        (1 + np.exp(self.steps_done / self.epsilon_decay))
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item() # 选择 Q 值最大的动作
        else:
            return random.randrange(self.action_dim) # 随机探索

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # 随机采样 Mini-batch [cite: 383]
        transitions = random.sample(self.memory, self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.FloatTensor(np.array(batch_state)).to(self.device)
        batch_action = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(self.device)
        batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)

        # --- Double DQN 核心逻辑 (Eq. 6) ---
        # 1. 使用 Policy Net 选动作
        current_q_values = self.policy_net(batch_state).gather(1, batch_action)
        next_actions = self.policy_net(batch_next_state).max(1)[1].unsqueeze(1)
        
        # 2. 使用 Target Net 算 Q 值
        next_q_values = self.target_net(batch_next_state).gather(1, next_actions)
        
        # 计算目标值: r + gamma * Q_target
        expected_q_values = batch_reward + (1 - batch_done) * self.gamma * next_q_values

        # 计算 Loss 并反向传播
        loss = self.loss_func(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 软更新 Target Network 参数 [cite: 387]
        # tau 设置为 0.005 (常用软更新系数)
        tau = 0.005
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)