from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_wordpiece_model(files=None, save_path=None):
    """
    训练WordPiece模型并保存

    参数：
        files(list): 训练数据文件列表
        save_path(str): 保存模型的路径
    """
    if files is None:
        files = [f"../../dataset/wiki.test.raw"]
    if save_path is None:
        save_path = "./models/tokenizer-wiki.json"
    # 创建一个空白的WordPiece tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    # 实例化WordPiece tokenizer的训练器
    trainer = WordPieceTrainer(
        special_tokens=["[UNK]","[CLK]","[SEP]","[PAD]","[MASK]"],
        min_frequency=1,
        show_progress=True,
        vocab_size=40000
    )

    # 定义预分词规则（这里使用空格切分）
    tokenizer.pre_tokenizer = Whitespace()

    # 加载数据集，训练tokenizer
    tokenizer.train(files, trainer)

    # 保存tokenizer
    tokenizer.save(save_path)

def wordpiece_tokenizer(save_path=None):
    """
    使用训练好的WordPiece tokenizer进行分词

    参数:
        save_path(str): 训练好的wordpiece tokenizer模型的保存路径
    """
    if save_path is None:
        save_path = "./models/tokenizer-wiki.json"
    
    # 加载tokenizer
    tokenizer = Tokenizer.from_pretrained(save_path)

    # 使用tokenizer对句子进行分词
    sentence = "我爱北京天安门，天安门上太阳升"
    output = tokenizer.encode(sentence)

    print("sentence: ",sentence)
    print("output.tokens: ",output.tokens)
    print("output.ids: ",output.ids)

if __name__ == "__main__":
    train_wordpiece_model()
    wordpiece_tokenizer()