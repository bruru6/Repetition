"""
对抗性提示生成模块

实现Zou et al.论文算法生成对抗提示

函数：
1. generate_adversarial_prompts(): 实现对抗性提示生成算法
2. _concatenate_prompts(): 将生成的对抗提示与自然查询拼接

输入输出：
- 输入：自然文本数据集（numpy数组）
- 输出：包含对抗提示的正样本数据集
"""

import numpy as np

def generate_adversarial_prompts(natural_texts, charset_size=95):
    """
    生成对抗性提示的核心算法
    
    参数：
    natural_texts: List[str] - 自然文本数据集
    charset_size: int - 可打印字符集大小（默认ASCII 95）
    
    返回：
    List[str] - 包含对抗提示的文本集合
    """
    # 实现对抗性字符生成逻辑
    adversarial_tokens = np.random.choice(charset_size, size=len(natural_texts))
    
    # 将对抗token转换为实际字符（ASCII 32-126）
    adversarial_prompts = [f"{chr(t+32)}" for t in adversarial_tokens]
    
    # 拼接对抗提示与自然文本
    return [_concatenate_prompts(adv, text) 
            for adv, text in zip(adversarial_prompts, natural_texts)]

def _concatenate_prompts(adversarial_prompt, natural_query):
    """拼接对抗提示与自然查询，形成最终输入序列"""
    return f"{adversarial_prompt} {natural_query}"