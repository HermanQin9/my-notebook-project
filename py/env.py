# env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pipeline import run_pipeline

class PipelineEnv(gym.Env):
    """
    强化学习环境，用于优化机器学习管道配置
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, df_train, config=None):
        super().__init__()
        self.df = df_train
        
        # 环境配置
        self.n_nodes = 7
        self.n_methods = 3
        self.n_combinations = self.n_nodes * self.n_methods
        
        # 环境参数
        self.initial_budget = config.get('budget', 100.0) if config else 100.0
        self.max_steps = config.get('max_steps', 50) if config else 50
        self.cost_weight = config.get('cost_weight', 0.01) if config else 0.01
        self.performance_bonus = config.get('performance_bonus', 10.0) if config else 10.0
        self.performance_threshold = config.get('performance_threshold', 0.1) if config else 0.1
        
        # 动作空间：扁平化为连续空间 [combination_choice, hp1, hp2]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 观察空间：配置one-hot编码 + 性能指标 + 预算信息
        obs_dim = self.n_nodes * self.n_methods + 4  # config + mae + cost + budget + step_ratio
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # 初始化状态
        self.reset()
    
    def _decode_action(self, action):
        """
        将连续动作解码为离散的 node, method, hyperparameters
        
        Args:
            action: numpy array [combination_choice, hp1, hp2]
        
        Returns:
            tuple: (node, method, hyperparameters)
        """
        # 将 action[0] 映射到组合索引
        combination_idx = int(action[0] * self.n_combinations)
        combination_idx = min(combination_idx, self.n_combinations - 1)
        
        # 解码节点和方法
        node = combination_idx // self.n_methods
        method = combination_idx % self.n_methods
        
        # 超参数
        hp = action[1:3]
        
        return node, method, hp
    
    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        
        # 重置环境状态
        self.budget = self.initial_budget
        self.step_count = 0
        self.config = {i: 0 for i in range(self.n_nodes)}
        
        # 获取初始性能
        try:
            mae, cost, *_ = run_pipeline(self.config, self.df)
        except Exception as e:
            print(f"Warning: run_pipeline failed during reset: {e}")
            mae, cost = 1.0, 0.0
        
        # 构建初始观察
        obs = self._build_observation(mae, cost)
        info = self._build_info(mae, cost, 0, 0, "reset")
        
        return obs, info
    
    def step(self, action):
        """执行一个动作步骤"""
        self.step_count += 1
        
        # 解码动作
        node, method, hp = self._decode_action(action)
        
        # 保存之前的配置
        prev_config = self.config.copy()
        prev_budget = self.budget
        
        # 更新配置
        self.config[node] = method
        
        # 检查配置一致性
        if not self._is_valid_config():
            # 无效配置，恢复并给予惩罚
            self.config = prev_config
            obs = self._build_observation(0.0, 0.0)
            reward = -1.0
            terminated = False
            truncated = self.step_count >= self.max_steps
            info = self._build_info(0.0, 0.0, node, method, "invalid_config")
            return obs, reward, terminated, truncated, info
        
        # 运行管道获取性能
        try:
            mae, cost, *_ = run_pipeline(self.config, self.df)
        except Exception as e:
            print(f"Warning: run_pipeline failed: {e}")
            # 给予高成本和低性能作为惩罚
            mae, cost = 10.0, 5.0
        
        # 检查预算约束
        if cost > self.budget:
            # 预算不足，恢复配置
            self.config = prev_config
            self.budget = prev_budget
            obs = self._build_observation(mae, cost)
            reward = -5.0  # 预算不足惩罚
            terminated = True
            truncated = False
            info = self._build_info(mae, cost, node, method, "budget_exceeded")
            return obs, reward, terminated, truncated, info
        
        # 更新预算
        self.budget -= cost
        
        # 计算奖励
        reward = self._calculate_reward(mae, cost)
        
        # 检查终止条件
        terminated = self.budget <= 0
        truncated = self.step_count >= self.max_steps
        
        # 构建观察和信息
        obs = self._build_observation(mae, cost)
        info = self._build_info(mae, cost, node, method, "normal")
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self, mae, cost):
        """计算奖励函数"""
        # 基础奖励：负的MAE（越小越好）
        reward = -mae
        
        # 成本惩罚
        reward -= self.cost_weight * cost
        
        # 性能奖励：如果性能很好，给额外奖励
        if mae < self.performance_threshold:
            reward += self.performance_bonus
        
        # 预算效率奖励：剩余预算比例
        budget_efficiency = self.budget / self.initial_budget
        reward += budget_efficiency * 0.1
        
        return reward
    
    def _build_observation(self, mae, cost):
        """构建观察向量"""
        # 配置的one-hot编码
        config_vector = np.zeros(self.n_nodes * self.n_methods, dtype=np.float32)
        for node, method in self.config.items():
            if 0 <= node < self.n_nodes and 0 <= method < self.n_methods:
                idx = node * self.n_methods + method
                config_vector[idx] = 1.0
        
        # 性能和预算信息
        step_ratio = self.step_count / self.max_steps
        performance_info = np.array([
            float(mae),
            float(cost),
            float(self.budget / self.initial_budget),  # 归一化预算
            float(step_ratio)
        ], dtype=np.float32)
        
        return np.concatenate([config_vector, performance_info])
    
    def _build_info(self, mae, cost, node, method, status):
        """构建信息字典"""
        return {
            "mae": float(mae),
            "cost": float(cost),
            "budget": float(self.budget),
            "budget_ratio": float(self.budget / self.initial_budget),
            "config": self.config.copy(),
            "node": int(node),
            "method": int(method),
            "step": self.step_count,
            "step_ratio": float(self.step_count / self.max_steps),
            "status": status
        }
    
    def _is_valid_config(self):
        """检查配置是否有效"""
        # 这里可以添加具体的配置验证逻辑
        # 例如：检查依赖关系、资源约束等
        
        # 简单检查：确保所有节点都有有效的方法
        for node, method in self.config.items():
            if not (0 <= method < self.n_methods):
                return False
        
        return True
    
    def render(self, mode="human"):
        """渲染环境状态"""
        if mode == "human":
            print(f"\n=== Pipeline Environment State ===")
            print(f"Step: {self.step_count}/{self.max_steps}")
            print(f"Budget: {self.budget:.2f}/{self.initial_budget:.2f}")
            print(f"Config: {self.config}")
            print(f"=====================================")
    
    def close(self):
        """清理资源"""
        pass
    
    def get_current_performance(self):
        """获取当前配置的性能"""
        try:
            mae, cost, *_ = run_pipeline(self.config, self.df)
            return mae, cost
        except Exception as e:
            print(f"Error getting current performance: {e}")
            return None, None
    
    def set_config(self, config):
        """手动设置配置（用于测试）"""
        if isinstance(config, dict) and len(config) == self.n_nodes:
            self.config = config.copy()
            return True
        return False