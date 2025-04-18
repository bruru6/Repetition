"""
数据加载与处理模块

实现功能：
1. 加载WikiText公开数据集
2. 执行训练集/验证集/测试集划分
3. 数据预处理（tokenization等）

参考论文：
- 数据集划分比例采用原文70%/15%/15%的比例
"""

from datasets import load_dataset
from sklearn.model_selection import train_test_split

def load_wikitext_dataset(dataset_name='wikitext', version='wikitext-103-v1'):
    """
    加载Hugging Face的WikiText数据集
    
    参数：
    dataset_name: str - 数据集名称（默认'wikitext'）
    version: str - 数据集版本（默认'wikitext-103-v1'）
    
    返回：
    DatasetDict - 包含完整数据集的对象
    """
    return load_dataset(dataset_name, version)

def create_train_test_split(full_dataset, test_size=0.3, random_state=42):
    """
    执行数据集划分（70%训练，30%临时）
    
    参数：
    full_dataset: Dataset - 完整数据集对象
    test_size: float - 测试集占比（默认0.3）
    random_state: int - 随机种子（默认42）
    
    返回：
    Tuple[Dataset, Dataset, Dataset] - (训练集, 验证集, 测试集)
    """
    # 首次划分：70%训练，30%临时
    train_data, temp_data = train_test_split(
        full_dataset['train'],
        test_size=test_size,
        random_state=random_state
    )
    
    # 二次划分：15%验证，15%测试
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=random_state
    )
    
    return train_data, val_data, test_data