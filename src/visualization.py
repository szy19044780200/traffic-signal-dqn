import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import os

class TrafficVisualizer:
    """
    交通仿真可视化模块
    """
    
    def __init__(self, env, training_history=None):
        self.env = env
        self.training_history = training_history or []
        
        # 创建结果目录
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('models'):
            os.makedirs('models')
    
    def plot_training_evaluation(self):
        """
        绘制训练曲线和评估结果
        """
        if not self.training_history:
            print("No training history available")
            return
        
        episodes = [h['episode'] for h in self.training_history]
        rewards = [h['reward'] for h in self.training_history]
        wait_times = [h['avg_wait_time'] for h in self.training_history]
        vehicles = [h['vehicles_passed'] for h in self.training_history]
        
        # 计算移动平均
        window = 10
        rewards_ma = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        wait_ma = [np.mean(wait_times[max(0, i-window):i+1]) for i in range(len(wait_times))]
        vehicles_ma = [np.mean(vehicles[max(0, i-window):i+1]) for i in range(len(vehicles))]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('DQN Training Evaluation', fontsize=16, fontweight='bold')
        
        # 图1：奖励曲线
        axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.3, label='Episode Reward')
        axes[0, 0].plot(episodes, rewards_ma, 'b-', linewidth=2, label='Moving Average (10)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 图2：平均等待时间
        axes[0, 1].plot(episodes, wait_times, 'r-', alpha=0.3, label='Episode Wait Time')
        axes[0, 1].plot(episodes, wait_ma, 'r-', linewidth=2, label='Moving Average (10)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Wait Time (steps)')
        axes[0, 1].set_title('Average Wait Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 图3：通过车数
        axes[1, 0].plot(episodes, vehicles, 'g-', alpha=0.3, label='Episode Vehicles')
        axes[1, 0].plot(episodes, vehicles_ma, 'g-', linewidth=2, label='Moving Average (10)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Vehicles Passed')
        axes[1, 0].set_title('Throughput')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 图4：统计信息
        axes[1, 1].axis('off')
        stats_text = f"""
        Training Statistics:
        ━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Total Episodes: {len(episodes)}
        
        Final Reward: {rewards[-1]:.2f}
        Best Reward: {max(rewards):.2f}
        
        Final Wait Time: {wait_times[-1]:.2f}
        Best Wait Time: {min(wait_times):.2f}
        
        Final Vehicles: {vehicles[-1]:.0f}
        Best Vehicles: {max(vehicles):.0f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('results/training_evaluation.png', dpi=100, bbox_inches='tight')
        print("Training evaluation plot saved to: results/training_evaluation.png")
        plt.close()
    
    def plot_detailed_comparison(self, results):
        """
        绘制详细的方法对比
        """
        methods = ['DQN', 'Fixed\nTiming', 'Sensor-\nbased', 'Random']
        wait_times = [
            results['dqn']['avg_wait_time'],
            results['fixed']['avg_wait_time'],
            results['sensor']['avg_wait_time'],
            results['random']['avg_wait_time']
        ]
        vehicles = [
            results['dqn']['vehicles_passed'],
            results['fixed']['vehicles_passed'],
            results['sensor']['vehicles_passed'],
            results['random']['vehicles_passed']
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Performance Comparison', fontsize=16, fontweight='bold')
        
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        
        # 图1：等待时间对比
        bars1 = axes[0].bar(methods, wait_times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Average Wait Time (steps)', fontsize=12)
        axes[0].set_title('Average Wait Time Comparison', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars1, wait_times):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom', fontweight='bold')
        
        # 图2：通过车数对比
        bars2 = axes[1].bar(methods, vehicles, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[1].set_ylabel('Vehicles Passed', fontsize=12)
        axes[1].set_title('Throughput Comparison', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars2, vehicles):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.0f}',
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/detailed_comparison.png', dpi=100, bbox_inches='tight')
        print("Detailed comparison plot saved to: results/detailed_comparison.png")
        plt.close()
    
    def static_visualization(self, agent, num_episodes=1):
        """
        静态可视化：绘制十字路口交通场景
        """
        fig, axes = plt.subplots(1, num_episodes, figsize=(8*num_episodes, 8))
        
        if num_episodes == 1:
            axes = [axes]
        
        for episode_idx in range(num_episodes):
            state = self.env.reset()
            
            for step in range(500):
                q_values = agent.main_network.predict(state[np.newaxis, :], verbose=0)
                action = np.argmax(q_values[0])
                state, _, done, _ = self.env.step(action)
                
                if done:
                    break
            
            self._draw_intersection(axes[episode_idx], step)
        
        plt.tight_layout()
        plt.savefig('results/traffic_animation.png', dpi=100, bbox_inches='tight')
        print("Traffic animation saved to: results/traffic_animation.png")
        plt.close()
    
    def _draw_intersection(self, ax, step):
        """
        绘制十字路口
        """
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 绘制道路
        ax.add_patch(Rectangle((2, 0), 6, 10, fill=True, facecolor='lightgray', edgecolor='black'))
        ax.add_patch(Rectangle((0, 2), 10, 6, fill=True, facecolor='lightgray', edgecolor='black'))
        
        # 绘制路口中心（黄色）
        ax.add_patch(Rectangle((3, 3), 4, 4, fill=True, facecolor='yellow', edgecolor='black', linewidth=2))
        
        # NS信号灯
        ns_color = 'green' if self.env.ns_light == 1 else 'red'
        ax.add_patch(Circle((1, 5), 0.3, color=ns_color))
        ax.add_patch(Circle((9, 5), 0.3, color=ns_color))
        
        # EW信号灯
        ew_color = 'green' if self.env.ew_light == 1 else 'red'
        ax.add_patch(Circle((5, 1), 0.3, color=ew_color))
        ax.add_patch(Circle((5, 9), 0.3, color=ew_color))
        
        # 绘制队列
        queue_n = min(len(self.env.queues['N']), 8)
        queue_s = min(len(self.env.queues['S']), 8)
        queue_e = min(len(self.env.queues['E']), 8)
        queue_w = min(len(self.env.queues['W']), 8)
        
        # North
        for i in range(queue_n):
            ax.add_patch(Rectangle((4, 6 + i*0.4), 0.3, 0.3, fill=True, facecolor='blue', edgecolor='darkblue'))
        
        # South
        for i in range(queue_s):
            ax.add_patch(Rectangle((5.7, 2.5 - i*0.4), 0.3, 0.3, fill=True, facecolor='blue', edgecolor='darkblue'))
        
        # East
        for i in range(queue_e):
            ax.add_patch(Rectangle((6 + i*0.4, 4), 0.3, 0.3, fill=True, facecolor='blue', edgecolor='darkblue'))
        
        # West
        for i in range(queue_w):
            ax.add_patch(Rectangle((2.5 - i*0.4, 5.7), 0.3, 0.3, fill=True, facecolor='blue', edgecolor='darkblue'))
        
        # 标题
        ax.set_title(f'Step: {step}\nQueues: N={queue_n} S={queue_s} E={queue_e} W={queue_w}',
                    fontsize=12, fontweight='bold')
        
        # 添加图例
        ax.text(0.5, 0.2, 'N→', fontsize=10)
        ax.text(9.5, 0.2, '←N', fontsize=10)
        ax.text(0.2, 5, '↓S', fontsize=10)
        ax.text(0.2, 9.7, '↑S', fontsize=10)
