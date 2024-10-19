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
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """
    合并词汇表中频率最高的词汇
    :param pair: 要合并的字符对
    :param vocab: 当前的词汇表
    :return: 合并指定字符对后的更新词汇表
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram +r'(?!\S)')
    for word in vocab:
        w_out = p.sub(''.join(pair),word)
        v_out[w_out] = vocab[word]
    return v_out

def get_tokens(vocab):
    """
    从当前词汇表生成标记
    :param vocab: 具有标记频率的词汇表
    :return: 具有频率的标记
    """
    tokens = collections.defaultdict(int)
    for word,freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens

def tokenize_word(word,tokenizer):
    tokens = []
    while word:
        found = False
        for i in range(len(word),0,-1):
            subword = word[:i]
            if subword in tokenizer:
                tokens.append(subword)
                word = word[i:]
                found = True
                break
        if not found:
            tokens.append(word[0])
            word = word[1:]
    return tokens

def main():
    # 示例词汇表
    vocab = {"c a r":5,"c a b b a g e":3,"t a b l e":1,"d e t c h":2,"c h a i r":5}
    print("===================================")
    print("BPE分词前的标记")
    bpe_tokens = get_tokens(vocab)
    print("标记 : {}".format(bpe_tokens))
    print("标记数 : {}".format(len(bpe_tokens)))
    print("===================================")

    vocab_size = 13
    i = 0
    while len(bpe_tokens) < vocab_size:
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs,key=pairs.get)
        vocab = merge_vocab(best,vocab)
        print("迭代 : {}".format(i))
        print("最佳字符对 : {}".format(best))
        bpe_tokens = get_tokens(vocab)
        print("标记 : {}".format(bpe_tokens))
        print("标记数 : {}".format(len(bpe_tokens)))
        print("===================================")
        i += 1
    word = "card"
    print(f"{word} 的分词结果:{tokenize_word(word,vocab)}")

if __name__ == "__main__":
    main()