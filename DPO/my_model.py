from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
)
import argparse

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:
    from transformers.deepspeed import is_deepspeed_zero3_enabled
import torch

"""
    模型读取
"""

def parse_args():
    parser = argparse.ArgumentParser(description='load_model')
    parser.add_argument('--model_path',type=str,default='/root/autodl-fs/TinyLlama/TinyLlama-1.1B-Chat-v0.6',help='模型地址')
    return parser.parse_args()

def load_model(args):
    """
    Args:
        model_path: 模型路径
    
    Return:
        model: 加载好的模型
        tokenizer: 模型对应的分词器
        ref_model: 参考模型
    """
    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code = True,
        torch_dtype = torch.float32,
        cache_dir = None
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.float32,
        load_in_4bit=True,
        load_in_8bit=False,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map='auto',
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
        )
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        load_in_8bit=False,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map='auto',
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    )
    return model, tokenizer, ref_model