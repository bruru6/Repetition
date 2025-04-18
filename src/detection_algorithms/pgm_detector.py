"""
基于概率图模型(PGM)的检测算法模块

实现功能：
1. 前向-后向算法计算边缘概率
2. 后验概率估计
3. 生成token级和序列级检测结果

参考公式：
联合概率：p(c|x) ∝ exp(Σ[(1-c_i)log(p0,i)+c_i log(p1,i)] - λΣ|c_{i+1}-c_i| - μΣc_i)
"""

import numpy as np

def pgm_based_detection(p0, p1, lambda_, mu):
    """
    前向-后向算法实现PGM检测
    
    参数：
    p0: np.ndarray - 正常分布概率 (n_tokens,)
    p1: np.ndarray - 对抗分布概率 (n_tokens,)
    lambda_: float - 平滑系数
    mu: float - 惩罚系数
    
    返回：
    np.ndarray - 各token属于对抗提示的后验概率
    """
    n = len(p0)
    # 初始化前向和后向消息
    forward = np.zeros((n, 2))
    backward = np.zeros((n, 2))
    
    # 前向传递
    forward[0] = [1, 1]  # 初始状态均匀分布
    for t in range(1, n):
        for c_t in [0, 1]:
            log_potential = (np.log(p0[t] if c_t==0 else p1[t]) - 
                            mu * c_t - 
                            lambda_ * abs(c_t - forward[t-1]))
            forward[t, c_t] = np.sum(np.exp(log_potential) * forward[t-1])
    
    # 后向传递
    backward[-1] = [1, 1]
    for t in range(n-2, -1, -1):
        for c_t in [0, 1]:
            log_potential = (np.log(p0[t+1] if backward[t+1]==0 else p1[t+1]) - 
                            mu * backward[t+1] - 
                            lambda_ * abs(backward[t+1] - c_t))
            backward[t, c_t] = np.sum(np.exp(log_potential) * backward[t+1])
    
    # 计算边缘概率
    posterior = forward * backward
    posterior /= np.sum(posterior, axis=1, keepdims=True)
    
    return posterior[:, 1]


def calculate_posterior_probability(posteriors):
    """计算全句最大后验概率"""
    return np.max(posteriors)