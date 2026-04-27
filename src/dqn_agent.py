import numpy as np
import random
from collections import deque
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class DoubleDQNAgent:
    """
    Double DQN智能体
    
    相比标准DQN的改进：
    1. 两个网络：主网络和目标网络
    2. 分离选择和评估，解决Q值高估
    3. 更稳定的训练
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # 超参数
        self.gamma = 0.95           # 折扣因子
        self.epsilon = 1.0          # 初始探索率
        self.epsilon_min = 0.01     # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # 神经网络
        self.main_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
    
    def _build_model(self):
        """构建神经网络"""
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def act(self, state):
        """选择动作（ε-贪心策略）"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.choice(self.action_size)
        else:
            # 利用：选择Q值最大的动作
            state_input = state[np.newaxis, :]
            q_values = self.main_network.predict(state_input, verbose=0)
            return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样小批量
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # Double DQN: 用主网络选择最优动作，用目标网络计算Q值
        # 步骤1：用主网络在next_state上选择最优动作
        next_q_main = self.main_network.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_main, axis=1)
        
        # 步骤2：用目标网络计算这些动作的Q值
        next_q_target = self.target_network.predict(next_states, verbose=0)
        next_q_values = next_q_target[np.arange(self.batch_size), next_actions]
        
        # 步骤3：计算Bellman目标
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 步骤4：预测当前状态的Q值
        q_values = self.main_network.predict(states, verbose=0)
        q_values[np.arange(self.batch_size), actions] = targets
        
        # 步骤5：训练主网络
        self.main_network.fit(states, q_values, epochs=1, verbose=0)
    
    def update_target_network(self):
        """同步目标网络权重到主网络"""
        self.target_network.set_weights(self.main_network.get_weights())
    
    def decay_epsilon(self):
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """保存模型"""
        self.main_network.save(filepath)
    
    def load_model(self, filepath):
        """加载模型"""
        self.main_network = keras.models.load_model(filepath)
        self.update_target_network()
