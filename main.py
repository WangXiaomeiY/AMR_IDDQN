import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
from env import AMRGridEnv
from agent import IDDQNAgent

# 贝塞尔曲线平滑算法 [cite: 358-364]
# === 将这段代码替换掉原来的 bezier_smooth 函数 ===

# 1. 提取关键拐点函数
def get_corners(path_points):
    if len(path_points) < 3: return np.array(path_points)
    
    # 始终保留起点
    corners = [path_points[0]]
    path_points = np.array(path_points)
    
    # 计算初始方向
    last_dir = path_points[1] - path_points[0]
    
    for i in range(1, len(path_points) - 1):
        # 计算当前这一步的方向
        curr_dir = path_points[i+1] - path_points[i]
        
        # 如果方向发生了变化（向量不相等），说明是拐点
        if not np.array_equal(curr_dir, last_dir):
            corners.append(path_points[i])
            last_dir = curr_dir
            
    # 始终保留终点
    corners.append(path_points[-1])
    return np.array(corners)

def bezier_smooth(path_points, num_insert=10):
    if len(path_points) < 3: return np.array(path_points)
    smoothed_path = []
    path_points = np.array(path_points)
    
    # 使用二阶贝塞尔曲线连接路径点
    for i in range(0, len(path_points)-2):
        p0 = path_points[i]
        p1 = path_points[i+1]
        p2 = path_points[i+2]
        
        # 在每两个点之间插值
        for t in np.linspace(0, 1, num_insert):
            # Eq. 17
            bt = (1-t)**2 * p0 + 2*t*(1-t) * p1 + t**2 * p2
            smoothed_path.append(bt)
            
    smoothed_path.append(path_points[-1])
    return np.array(smoothed_path)

if __name__ == "__main__":
    env = AMRGridEnv()
    agent = IDDQNAgent(state_dim=4, action_dim=8)
    
    episodes = 1000 # 训练轮数
    rewards_history = []
    
    print("开始训练...")
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        path = [state[:2]]
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            path.append(state[:2])
            
            if done:
                break
        
        rewards_history.append(total_reward)
        if ep % 10 == 0:
            print(f"Episode: {ep}, Reward: {total_reward:.2f}, Epsilon steps: {agent.steps_done}")

    # --- 绘图部分 ---
    print("训练结束，正在绘图...")
    
    # 1. 绘制最终路径
    plt.figure(figsize=(8, 8))
    # 画障碍物
    for obs in env.obstacles:
        plt.plot(obs[0], obs[1], 'ks', markersize=15) # 黑色方块代表障碍物
    
    raw_path = np.array(path)
    smooth_path = bezier_smooth(raw_path) # 应用贝塞尔平滑
    
    plt.plot(smooth_path[:, 0], smooth_path[:, 1], color='blue', linestyle='-', linewidth=2, label='Smoothed Path (Bezier)')
    plt.plot(raw_path[:, 0], raw_path[:, 1], color='red', linestyle='--', linewidth=1.5, label='IDDQN Path (Raw)')
    
    plt.plot(env.start_pos[0], env.start_pos[1], 'go', markersize=10, label='Start')
    plt.plot(env.target_pos[0], env.target_pos[1], 'r*', markersize=15, label='Target')
    
    plt.legend()
    plt.grid(True)
    plt.title("Path Planning Result")
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.show()
    
    # 2. 绘制奖励曲线
    plt.figure()
    plt.plot(rewards_history)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()