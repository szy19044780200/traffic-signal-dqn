import numpy as np
import random
from collections import deque

class TrafficEnvironment:
    """
    十字路口交通仿真环境
    
    4条道路: North, South, East, West
    状态空间: 11维
    动作空间: 2个离散动作 {0: 保持, 1: 切换}
    """
    
    def __init__(self, arrival_rates=None, max_episode_steps=1000, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 车辆到达率（Poisson过程）
        self.arrival_rates = arrival_rates or {
            'N': 0.15,
            'S': 0.15,
            'E': 0.20,
            'W': 0.20
        }
        
        # 初始化4条道路的车队
        self.queues = {'N': deque(), 'S': deque(), 'E': deque(), 'W': deque()}
        
        # 信号灯状态
        self.ns_light = 1  # 1: NS绿灯, 0: NS红灯
        self.ew_light = 0  # 1: EW绿灯, 0: EW红灯
        
        # 每条车道的等待时间记录
        self.wait_times = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        
        # 统计信息
        self.vehicles_passed = 0
        self.total_wait_time = 0
        self.steps = 0
        self.max_episode_steps = max_episode_steps
    
    def reset(self):
        """重置环境到初始状态"""
        self.queues = {'N': deque(), 'S': deque(), 'E': deque(), 'W': deque()}
        self.ns_light = 1
        self.ew_light = 0
        self.wait_times = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        self.vehicles_passed = 0
        self.total_wait_time = 0
        self.steps = 0
        return self._get_state()
    
    def step(self, action):
        """执行一步动作"""
        # 1. 执行动作（切换或保持信号灯）
        if action == 1:
            self.ns_light = 1 - self.ns_light
            self.ew_light = 1 - self.ew_light
        
        # 2. 生成新车辆
        self._generate_vehicles()
        
        # 3. 计算等待时间（在车辆通过前）
        wait_time = self._calculate_wait_time()
        
        # 4. 车辆通过（根据绿灯状态）
        vehicles_passed = self._process_vehicles()
        
        # 5. 计算奖励
        reward = vehicles_passed * 1.0 - 0.01 * wait_time
        
        # 6. 更新统计信息
        self.vehicles_passed += vehicles_passed
        self.total_wait_time += wait_time
        self.steps += 1
        
        # 7. 检查是否结束
        done = self.steps >= self.max_episode_steps
        
        return self._get_state(), reward, done, {}
    
    def _generate_vehicles(self):
        """根据Poisson过程生成新车辆"""
        for direction in ['N', 'S', 'E', 'W']:
            # 根据到达率，决定是否生成新车辆
            if np.random.random() < self.arrival_rates[direction]:
                # 新车辆的等待时间初始为0
                self.queues[direction].append(0)
    
    def _process_vehicles(self):
        """处理车辆通过（根据绿灯状态）"""
        vehicles_passed = 0
        
        # NS方向：如果NS灯为绿
        if self.ns_light == 1:
            # 最多通过3辆车（物理限制）
            pass_count = min(3, len(self.queues['N']))
            for _ in range(pass_count):
                self.queues['N'].popleft()
                vehicles_passed += 1
            
            pass_count = min(3, len(self.queues['S']))
            for _ in range(pass_count):
                self.queues['S'].popleft()
                vehicles_passed += 1
        
        # EW方向：如果EW灯为绿
        if self.ew_light == 1:
            pass_count = min(3, len(self.queues['E']))
            for _ in range(pass_count):
                self.queues['E'].popleft()
                vehicles_passed += 1
            
            pass_count = min(3, len(self.queues['W']))
            for _ in range(pass_count):
                self.queues['W'].popleft()
                vehicles_passed += 1
        
        return vehicles_passed
    
    def _calculate_wait_time(self):
        """计算所有车辆的平均等待时间"""
        total_wait = 0
        total_vehicles = 0
        
        for direction in ['N', 'S', 'E', 'W']:
            queue_length = len(self.queues[direction])
            total_wait += queue_length
            total_vehicles += queue_length
        
        return total_wait if total_vehicles > 0 else 0
    
    def _get_state(self):
        """获取当前状态（11维向量）"""
        # 队列长度（4维）
        queue_n = min(len(self.queues['N']), 50) / 50.0
        queue_s = min(len(self.queues['S']), 50) / 50.0
        queue_e = min(len(self.queues['E']), 50) / 50.0
        queue_w = min(len(self.queues['W']), 50) / 50.0
        
        # 等待时间（4维，归一化）
        wait_n = min(self.wait_times['N'], 100) / 100.0
        wait_s = min(self.wait_times['S'], 100) / 100.0
        wait_e = min(self.wait_times['E'], 100) / 100.0
        wait_w = min(self.wait_times['W'], 100) / 100.0
        
        # 信号灯状态（2维）
        ns_light = float(self.ns_light)
        ew_light = float(self.ew_light)
        
        # 时间步（1维，归一化）
        time_step = self.steps / self.max_episode_steps
        
        state = np.array([
            queue_n, queue_s, queue_e, queue_w,
            wait_n, wait_s, wait_e, wait_w,
            ns_light, ew_light,
            time_step
        ], dtype=np.float32)
        
        return state
