from datasets import load_dataset

"""
    数据集加载
"""

def load_data():
    """
    加载偏好对齐数据集
    Args:
        dataset_name: 数据集名称
        ratio: 验证集占总数据集比例
    
    Returns:
        train_data: 偏好对齐训练样本
        dev_data: 偏好对齐验证样本
    """
    dataset_name="/root/code/LLM_learning/DPO/data"
    ratio=0.1
    data_zh = load_dataset(path=dataset_name)
    data_all = data_zh['train'].train_test_split(ratio)
    train_data = data_all["train"]
    dev_data = data_all["test"]
    print(len(train_data))
    print(len(dev_data))
    return train_data,dev_data