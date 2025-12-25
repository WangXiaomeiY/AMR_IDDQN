# AMR IDDQN - Autonomous Mobile Robot Navigation using Deep Q-Network

## 项目描述

这是一个基于改进型深度Q网络（IDDQN）的自主移动机器人（AMR）路径规划与导航系统。该项目使用强化学习算法训练机器人在网格环境中自主规划路径、避开障碍物，到达目标位置。

## 核心特性

- **IDDQN 算法实现**：改进的双网络（Policy Network & Target Network）深度Q学习算法
- **自适应 Epsilon 贪心策略**：动态调整探索-利用平衡
- **经验回放机制**：优先采样重要的训练数据
- **路径平滑**：使用贝塞尔曲线对生成的路径进行平滑处理
- **网格环境模拟**：支持随机障碍物生成与碰撞检测

## 项目结构

```
AMR_IDDQN/
├── main.py          # 主训练脚本
├── agent.py         # IDDQN 智能体实现
├── env.py           # 网格环境（基于 OpenAI Gym）
├── model.py         # 神经网络模型定义
├── requirements.txt # 项目依赖
└── README.md        # 项目说明文档
```

## 主要模块

### Agent (agent.py)
- **IDDQNAgent** 类：实现强化学习智能体
  - 策略网络与目标网络管理
  - Q值选择与动作决策
  - 经验回放与批量训练

### Environment (env.py)
- **AMRGridEnv** 类：30×30 网格环境
  - 支持8方向移动（上下左右及对角线）
  - 动态障碍物生成
  - 距离目标与碰撞奖励设计

### Model (model.py)
- **IDDQN** 网络：3层隐藏层(128单元)的多层感知器
  - 输入：状态向量(4维：robot_x, robot_y, 目标距离, 障碍物距离)
  - 输出：8个动作的Q值

### Main (main.py)
- 训练循环控制
- 路径规划与可视化
- 贝塞尔曲线平滑算法

## 环境要求

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- OpenAI Gym

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python main.py
```

训练过程会：
1. 初始化 IDDQN 智能体
2. 在网格环境中进行多轮探索
3. 实时绘制学习曲线和路径规划结果
4. 保存最优模型

## 算法参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 0.0025 | Adam 优化器学习率 |
| 折扣因子 γ | 0.9 | 未来奖励衰减系数 |
| 经验池容量 | 10,000 | 回放池最大容量 |
| 批处理大小 | 64 | 每次训练的样本数 |
| ε 初始值 | 0.9 | 初始探索率 |
| ε 最终值 | 0.01 | 最终探索率 |
| 隐藏层维度 | 128 | 神经网络隐藏层单元数 |

## 性能指标

- **目标到达率**：成功导航到目标位置的比例
- **平均步数**：到达目标所需的平均步数
- **碰撞次数**：训练过程中的碰撞统计

## 文献参考

本项目参考以下研究：
- Deep Q-Networks (DQN) - Mnih et al., 2015
- Double DQN - van Hasselt et al., 2016
- 机器人路径规划与障碍物避免

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过项目 Issue 页面联系。
