import collections
import re

def get_stats(vocab):
    """
    计算词汇表中相邻字符对的频率
    :param vocab: 具有标记频率的词汇表
    :return: 相邻字符对的频率
    """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) -1):
            pairs(symbols[i],symbols[i+1]) += freq
    return pairs