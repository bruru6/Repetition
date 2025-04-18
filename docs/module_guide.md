# 核心模块技术文档

## 数据准备模块

### 对抗样本生成流程

- 输入：原始文本序列
- 输出：对抗文本及扰动位置标记
- 核心方法：`AdversarialGenerator.generate()` 实现论文3.2节的替换策略

## 语言模型模块

### 概率计算方法

- 双分布概率估计：`ProbabilityCalculator` 同时计算正常(p0)和对抗(p1)分布
- 概率对齐：采用论文4.1节的温度缩放方法

## PGM检测算法

### 数学原理

$$
P(c|x) \propto \exp\left(\sum_{i} [(1-c_i)\log p_{0,i} + c_i\log p_{1,i}] - \lambda\sum |c_{i+1}-c_i| - \mu\sum c_i\right)
$$

### 代码实现

- `pgm_based_detection()` 实现前向-后向算法
- `calculate_posterior_probability()` 计算序列级检测结果

## 接口规范

| 模块     | 输入格式          | 输出格式                       |
| -------- | ----------------- | ------------------------------ |
| 数据准备 | text: str         | (adv_text: str, mask: ndarray) |
| 语言模型 | tokens: List[str] | (p0: ndarray, p1: ndarray)     |
| 检测算法 | p0, p1, λ, μ    | posterior: ndarray             |

> 详细算法实现参考论文第4章第2节
