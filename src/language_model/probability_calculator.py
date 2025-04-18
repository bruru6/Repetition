"""
语言模型概率计算模块

实现功能：
1. 计算每个token的条件概率p_LLM
2. 处理首token概率修正
3. 计算对抗分布均匀概率

参考公式：
- p0,i = softmax(logits)[x_i]
- p1,i = 1/|Σ_printable| (默认95)
"""

import torch
import torch.nn.functional as F

def calculate_token_probabilities(model, tokenizer, input_text):
    """
    计算输入文本的token级概率分布
    
    参数：
    model: transformers.PreTrainedModel - 加载的预训练模型
    tokenizer: transformers.PreTrainedTokenizer - 对应的tokenizer
    input_text: str - 输入文本
    
    返回：
    Tuple[torch.Tensor, torch.Tensor] - (正常分布概率p0, 对抗分布概率p1)
    """
    # Tokenize输入文本
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids
    
    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    
    # 计算正常分布概率
    probs = F.softmax(logits, dim=-1)
    p0 = torch.gather(probs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze()
    
    # 生成对抗分布概率（均匀分布）
    vocab_size = tokenizer.vocab_size
    p1 = torch.ones_like(p0) / 95  # 根据论文设置可打印字符数
    
    # 首token概率修正
    p0[0] = p1[0]
    
    return p0, p1