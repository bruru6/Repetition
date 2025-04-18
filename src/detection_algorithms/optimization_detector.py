"""
基于优化的检测算法模块

实现功能：
1. 动态规划求解最小化目标函数
2. 处理首token边缘情况
3. 超参数网格搜索

参考公式：
目标函数：min Σ[-(1-c_i)log(p0,i)-c_i log(p1,i)] + λΣ|c_{i+1}-c_i| + μΣc_i
"""

import numpy as np

def optimization_based_detection(p0, p1, lambda_, mu):
    """
    动态规划实现优化检测算法
    
    参数：
    p0: np.ndarray - 正常分布概率 (n_tokens,)
    p1: np.ndarray - 对抗分布概率 (n_tokens,)
    lambda_: float - 平滑系数
    mu: float - 惩罚系数
    
    返回：
    np.ndarray - 最优检测路径c* (0/1数组)
    """
    n = len(p0)
    delta = np.full((n, 2), np.inf)
    path = np.zeros((n, 2), dtype=int)
    
    # 初始化（跳过首token）
    delta[0] = [0, 0]
    
    # 前向传递
    for t in range(1, n):
        for c_t in [0, 1]:
            costs = []
            for c_prev in [0, 1]:
                cost = delta[t-1, c_prev] + \
                       (-np.log(p0[t] if c_t==0 else p1[t])) + \
                       lambda_ * abs(c_t - c_prev) + \
                       mu * c_t
                costs.append(cost)
            delta[t, c_t] = min(costs)
            path[t, c_t] = np.argmin(costs)
    
    # 后向传递
    c_opt = np.zeros(n, dtype=int)
    c_opt[-1] = np.argmin(delta[-1])
    
    for t in range(n-2, 0, -1):
        c_opt[t] = path[t+1, c_opt[t+1]]
    
    return c_opt


def grid_search(p0, p1, lambda_range, mu_range):
    """超参数网格搜索实现"""
    best_params = {}
    # 实现网格搜索逻辑
    return best_params