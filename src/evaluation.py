import numpy as np
from src.environment import TrafficEnvironment

class Evaluator:
    """
    评估框架：对比不同方法的性能
    """
    
    def __init__(self, env):
        self.env = env
    
    def evaluate_dqn(self, agent, episodes=50):
        """
        评估DQN智能体
        """
        wait_times = []
        vehicles_list = []
        
        for _ in range(episodes):
            state = self.env.reset()
            
            for step in range(1000):
                # 使用贪心策略（不探索）
                q_values = agent.main_network.predict(state[np.newaxis, :], verbose=0)
                action = np.argmax(q_values[0])
                
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                
                if done:
                    break
            
            if self.env.vehicles_passed > 0:
                avg_wait = self.env.total_wait_time / self.env.vehicles_passed
            else:
                avg_wait = 0
            
            wait_times.append(avg_wait)
            vehicles_list.append(self.env.vehicles_passed)
        
        return {
            'avg_wait_time': np.mean(wait_times),
            'vehicles_passed': np.mean(vehicles_list)
        }
    
    def evaluate_fixed_timing(self, episodes=50):
        """
        评估定时控制（基准方法1）
        每30步切换一次信号灯
        """
        wait_times = []
        vehicles_list = []
        
        for _ in range(episodes):
            state = self.env.reset()
            
            for step in range(1000):
                # 每30步切换一次信号灯
                action = 1 if step % 30 == 0 and step > 0 else 0
                
                state, _, done, _ = self.env.step(action)
                
                if done:
                    break
            
            if self.env.vehicles_passed > 0:
                avg_wait = self.env.total_wait_time / self.env.vehicles_passed
            else:
                avg_wait = 0
            
            wait_times.append(avg_wait)
            vehicles_list.append(self.env.vehicles_passed)
        
        return {
            'avg_wait_time': np.mean(wait_times),
            'vehicles_passed': np.mean(vehicles_list)
        }
    
    def evaluate_sensor_based(self, episodes=50):
        """
        评估感应控制（基准方法2）
        根据队列长度动态调整信号灯
        """
        wait_times = []
        vehicles_list = []
        
        for _ in range(episodes):
            state = self.env.reset()
            last_switch = 0
            
            for step in range(1000):
                # 获取各方向队列长度
                queue_n = len(self.env.queues['N'])
                queue_s = len(self.env.queues['S'])
                queue_e = len(self.env.queues['E'])
                queue_w = len(self.env.queues['W'])
                
                ns_queue = queue_n + queue_s
                ew_queue = queue_e + queue_w
                
                # 如果垂直方向拥堵超过水平方向且距上次切换超过10步
                if ns_queue > ew_queue + 5 and step - last_switch > 10:
                    if self.env.ew_light == 1:  # 如果EW灯是绿
                        action = 1  # 切换到NS
                        last_switch = step
                    else:
                        action = 0
                # 如果水平方向拥堵超过垂直方向且距上次切换超过10步
                elif ew_queue > ns_queue + 5 and step - last_switch > 10:
                    if self.env.ns_light == 1:  # 如果NS灯是绿
                        action = 1  # 切换到EW
                        last_switch = step
                    else:
                        action = 0
                else:
                    action = 0  # 保持
                
                state, _, done, _ = self.env.step(action)
                
                if done:
                    break
            
            if self.env.vehicles_passed > 0:
                avg_wait = self.env.total_wait_time / self.env.vehicles_passed
            else:
                avg_wait = 0
            
            wait_times.append(avg_wait)
            vehicles_list.append(self.env.vehicles_passed)
        
        return {
            'avg_wait_time': np.mean(wait_times),
            'vehicles_passed': np.mean(vehicles_list)
        }
    
    def evaluate_random(self, episodes=50):
        """
        评估随机控制（基准方法3）
        随机选择动作
        """
        wait_times = []
        vehicles_list = []
        
        for _ in range(episodes):
            state = self.env.reset()
            
            for step in range(1000):
                # 随机选择动作
                action = np.random.choice([0, 1])
                
                state, _, done, _ = self.env.step(action)
                
                if done:
                    break
            
            if self.env.vehicles_passed > 0:
                avg_wait = self.env.total_wait_time / self.env.vehicles_passed
            else:
                avg_wait = 0
            
            wait_times.append(avg_wait)
            vehicles_list.append(self.env.vehicles_passed)
        
        return {
            'avg_wait_time': np.mean(wait_times),
            'vehicles_passed': np.mean(vehicles_list)
        }
