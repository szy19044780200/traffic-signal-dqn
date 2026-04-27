import numpy as np
import json
from src.environment import TrafficEnvironment
from src.dqn_agent import DoubleDQNAgent
from src.evaluation import Evaluator
from src.visualization import TrafficVisualizer

def train_dqn(episodes=200):
    """
    训练DQN智能体
    """
    print("=" * 70)
    print("基于深度Q学习的交通信号灯控制仿真系统")
    print("=" * 70)
    print()
    
    # 创建环境
    env = TrafficEnvironment(
        arrival_rates={'N': 0.15, 'S': 0.15, 'E': 0.20, 'W': 0.20},
        max_episode_steps=1000,
        seed=42
    )
    
    # 创建DQN智能体
    agent = DoubleDQNAgent(state_size=11, action_size=2, learning_rate=0.001)
    
    training_history = []
    
    print(f"开始训练 {episodes} 轮次...")
    print()
    print(f"{'Episode':<10} {'Reward':<15} {'Avg Wait':<15} {'Vehicles':<15} {'Epsilon':<10}")
    print("-" * 65)
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            # 选择动作（ε-贪心）
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 经验回放训练
            agent.replay()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 每10轮同步目标网络
        if (episode + 1) % 10 == 0:
            agent.update_target_network()
        
        # 计算平均等待时间
        avg_wait = env.total_wait_time / max(env.vehicles_passed, 1) if env.vehicles_passed > 0 else 0
        
        # 记录训练历史
        training_history.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'epsilon': agent.epsilon,
            'avg_wait_time': avg_wait,
            'vehicles_passed': env.vehicles_passed
        })
        
        # 每20轮打印一次
        if (episode + 1) % 20 == 0:
            print(f"{episode + 1:<10} {episode_reward:<15.2f} {avg_wait:<15.2f} {env.vehicles_passed:<15} {agent.epsilon:<10.4f}")
    
    print("-" * 65)
    print("\n训练完成！")
    
    # 保存模型
    agent.save_model('models/dqn_model.h5')
    print(f"模型已保存到: models/dqn_model.h5")
    
    return agent, env, training_history

def evaluate_all_methods(agent, episodes=50):
    """
    评估所有方法的性能
    """
    print()
    print("=" * 70)
    print("开始评估和对比...")
    print("=" * 70)
    print()
    
    env = TrafficEnvironment(max_episode_steps=1000, seed=42)
    evaluator = Evaluator(env)
    
    print(f"{'Method':<20} {'Avg Wait Time':<20} {'Vehicles Passed':<20}")
    print("-" * 60)
    
    # 1. 评估DQN
    print("1. 评估 DQN 智能体...")
    dqn_results = evaluator.evaluate_dqn(agent, episodes=episodes)
    print(f"{'DQN':<20} {dqn_results['avg_wait_time']:<20.2f} {dqn_results['vehicles_passed']:<20.1f}")
    
    # 2. 评估定时控制
    print("2. 评估定时控制...")
    fixed_results = evaluator.evaluate_fixed_timing(episodes=episodes)
    print(f"{'Fixed Timing':<20} {fixed_results['avg_wait_time']:<20.2f} {fixed_results['vehicles_passed']:<20.1f}")
    
    # 3. 评估感应控制
    print("3. 评估感应控制...")
    sensor_results = evaluator.evaluate_sensor_based(episodes=episodes)
    print(f"{'Sensor-based':<20} {sensor_results['avg_wait_time']:<20.2f} {sensor_results['vehicles_passed']:<20.1f}")
    
    # 4. 评估随机控制
    print("4. 评估随机控制...")
    random_results = evaluator.evaluate_random(episodes=episodes)
    print(f"{'Random':<20} {random_results['avg_wait_time']:<20.2f} {random_results['vehicles_passed']:<20.1f}")
    
    print("-" * 60)
    
    # 计算改进
    improvement = (fixed_results['avg_wait_time'] - dqn_results['avg_wait_time']) / fixed_results['avg_wait_time'] * 100
    print(f"\nDQN相比定时控制改进: {improvement:.1f}%")
    
    return {
        'dqn': dqn_results,
        'fixed': fixed_results,
        'sensor': sensor_results,
        'random': random_results
    }

def main():
    """
    主函数：训练、评估、可视化
    """
    # 1. 训练DQN
    agent, env, training_history = train_dqn(episodes=200)
    
    # 2. 评估所有方法
    results = evaluate_all_methods(agent, episodes=50)
    
    # 3. 可视化
    print()
    print("=" * 70)
    print("生成可视化图表...")
    print("=" * 70)
    
    visualizer = TrafficVisualizer(env, training_history)
    visualizer.plot_training_evaluation()
    visualizer.plot_detailed_comparison(results)
    
    # 4. 保存结果总结
    summary = {
        'training_episodes': 200,
        'evaluation_episodes': 50,
        'results': {
            'dqn': {
                'avg_wait_time': float(results['dqn']['avg_wait_time']),
                'vehicles_passed': float(results['dqn']['vehicles_passed']),
            },
            'fixed': {
                'avg_wait_time': float(results['fixed']['avg_wait_time']),
                'vehicles_passed': float(results['fixed']['vehicles_passed']),
            },
            'sensor': {
                'avg_wait_time': float(results['sensor']['avg_wait_time']),
                'vehicles_passed': float(results['sensor']['vehicles_passed']),
            },
            'random': {
                'avg_wait_time': float(results['random']['avg_wait_time']),
                'vehicles_passed': float(results['random']['vehicles_passed']),
            }
        }
    }
    
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n所有任务完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
