import numpy as np
import gym
from gym import spaces

class AMRGridEnv(gym.Env):
    def __init__(self):
        super(AMRGridEnv, self).__init__()
        self.grid_size = 30
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(4,), dtype=np.float32)
        
        self.target_pos = np.array([28, 28])
        self.start_pos = np.array([2, 2])
        
        self.obstacles = [] # 这里先留空，去 reset 里生成
        self.safe_dist = 1 # 安全距离

    def reset(self):
        # 1. 重置机器人位置
        self.agent_pos = self.start_pos.astype(float)
        self.steps = 0
        
        # 2. === 随机生成障碍物的核心代码 ===
        self.obstacles = []
        num_obstacles = 10  # 这里决定生成多少个障碍物，你可以改这个数字
        
        for _ in range(num_obstacles):
            # 随机生成 x, y 坐标 (范围 0 到 29)
            # 使用 while 循环是为了防止障碍物生成在起点或终点上，导致游戏还没开始就结束了
            while True:
                pos = np.random.randint(0, self.grid_size, size=2)
                
                # 检查: 不要在起点，也不要在终点
                if not np.array_equal(pos, self.start_pos) and \
                   not np.array_equal(pos, self.target_pos):
                    self.obstacles.append(pos)
                    break # 位置合法，跳出循环生成下一个
                    
        return self._get_obs()

    def _get_obs(self):
        # 计算 TD (To Target Distance)
        td = np.linalg.norm(self.agent_pos - self.target_pos)
        # 计算 OD (Obstacle Distance) - 取最近障碍物的距离
        od = min([np.linalg.norm(self.agent_pos - obs) for obs in self.obstacles])
        return np.array([self.agent_pos[0], self.agent_pos[1], td, od], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        # 8个动作对应的坐标变化 (上, 下, 左, 右, 右上, 右下, 左上, 左下) [cite: 247-254]
        # 步长 grid = 1
        moves = [
            [0, 1], [0, -1], [-1, 0], [1, 0],   # 上下左右
            [1, 1], [1, -1], [-1, 1], [-1, -1]  # 对角线
        ]
        
        move = np.array(moves[action])
        prev_pos = self.agent_pos.copy()
        self.agent_pos += move
        
        # 边界限制 (防止跑出地图)
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)
        
        # --- 奖励函数设计 (Eq. 9) ---
        r1 = r2 = r3 = r4 = 0
        
        dist_to_target = np.linalg.norm(self.agent_pos - self.target_pos)
        min_obs_dist = min([np.linalg.norm(self.agent_pos - obs) for obs in self.obstacles])

        # 1. 目标奖励 r1 (Eq. 10)
        done = False
        if dist_to_target <= np.sqrt(2)/2: # 进入目标区域
            r1 = 100 # 给一个大的正奖励
            done = True
        
        # 2. 距离引导奖励 r2 (Eq. 11) - 引导机器人靠近目标
        r2 = -0.1 * dist_to_target 
        
        # 3. 边界惩罚 r3 (Eq. 12)
        if (self.agent_pos == 0).any() or (self.agent_pos == self.grid_size).any():
            r3 = -10
            
        # 4. 障碍物碰撞奖励 r4 (Eq. 13)
        if min_obs_dist < self.safe_dist:
            r4 = -50 # 碰撞惩罚
        
        # 综合奖励
        reward = r1 + r2 + r3 + r4
        
        # 限制最大步数，防止死循环
        if self.steps > 200:
            done = True
            
        return self._get_obs(), reward, done, {}