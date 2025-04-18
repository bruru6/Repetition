"""
超参数配置模块

实现功能：
1. 管理优化检测器超参数范围
2. 存储模型及数据集默认路径

参考论文：
- Zou et al. 中附录表5的超参数搜索范围
"""

# 优化检测器参数范围
OPTIMIZATION_PARAMS = {
    # 平滑项系数 λ 的对数空间采样范围
    'lambda_range': {
        'start': 0.2,
        'end': 2000,
        'num_samples': 41,
        'scale': 'log'
    },
    
    # 惩罚项系数 μ 的线性采样范围
    'mu_range': {
        'start': -5,
        'end': 5,
        'step': 0.5
    }
}

# 默认模型配置
MODEL_CONFIG = {
    'model_name': 'gpt2',
    'revision': '124M',
    'cache_dir': './models'
}

# 数据集默认路径
DATASET_PATHS = {
    'wikitext': './data/wikitext',
    'adversarial': './data/adversarial'
}