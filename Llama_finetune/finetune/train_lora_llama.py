# -*- coding: utf-8 -*-

import logging.config
import os
import argparse
from typing import List, Dict, Optional

import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainingArguments,
    Trainer,
    LlamaForCausalLM
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from collections import Counter

_compute_dype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

logger = logging.getLogger("train_model")

def parse_args():
    parser = argparse.ArgumentParser(description='llama2-7B QLoRA')
    parser.add_argument('--train_args_json',type=str,default='./llama2-7B_LoRA.json',help='TrainingArguments的json文件')
    parser.add_argument('--model_name_or_path',type=str,default='',help='模型id或local path')
    parser.add_argument('--train_data_path',type=str,default='../data/train.jsonl',help='训练数据集路径')
    parser.add_argument('--eval_data_path',type=str,default='../data/dev.jsonl',help='验证数据集路径')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_input_length',type=int,default=1024,help='instruction + input的最大长度')
    parser.add_argument('--max_output_length',type=int,default=1024,help='output的最大长度')
    parser.add_argument('--lora_rank',type=int,default=4,help='lora rank')
    parser.add_argument('--lora_dim',type=int,default=8,help='')
    parser.add_argument('--lora_alpha',type=int,default=32,help='lora alpha')
    parser.add_argument('--lora_dropout',type=float,default=0.05,help='lora dropout')
    parser.add_argument('--lora_module_name',type=list,default=["q_proj","v_proj"],help='')
    parser.add_argument('--resume_from_checkpoint',type=str,default=None,help='恢复训练的checkpoint路径')
    parser.add_argument('--prompt_text',type=str,default='',help='统一添加在所有数据前的指令文本')
    parser.add_argument('--compute_dtype', type=str, default='fp16', choices=['fp32','fp16','bf16'],help='计算数据类型')
    return parser.parse_args()

def tokenize_func(example, tokenizer, global_args, ignore_label_id=-100):
    """单样本tokenize处理"""
    # 加入global_prompt和该样本的instruction到question中
    question = global_args.prompt_text + example["instruction"]
    if example.get('input',None):
        if example['input'].strip():
            # 如果该样本存在input，换行加入到question中
            question += f'''\n{example['input']}'''
    answer = example['output']
    q_ids = tokenizer.encode(text=question,add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer,add_special_tokens=False)
    # 如果question或者answer长度超标就截断
    if len(q_ids) > global_args.max_input_length -2:
        q_ids = q_ids[:global_args.max_input_length - 2]
    if len(a_ids) > global_args.max_input_length -2:
        a_ids = a_ids[:global_args.max_input_length - 2]
    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids,a_ids)    
    question_length = len(q_ids) + 2
    labels = [ignore_label_id] * question_length + input_ids[question_length:]
    return {'input_ids': input_ids, 'labels': labels}

def get_dataset(data_path, tokenizer, global_args):
    """读取本地数据文件，并tokenize，shuffle，返回datasets.dataset"""
    data = load_dataset('json',data_files=data_path)
    column_names = data['train'].column_names
    dataset = data['train'].map(lambda example: tokenize_func(example,tokenizer,global_args), batched=False,remove_columns=column_names)
    dataset = dataset.shuffle(seed=global_args.seed)
    dataset = dataset.flatten_indices()
    return dataset

class DataCollatorForLlama2:
    def __init__(self,
                 pad_token_id: int,
                 max_length: int = 2048,
                 ignore_label_id: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length
    
    def __call__(self,batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        """根据batch最大长度做padding"""
        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list)
        input_ids, labels = [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d
            ids = d['input_ids'] + [self.pad_token_id] * pad_len
            label = d["labels"] + [self.ignore_label_id] * pad_len
            if batch_max_len > self.max_length:
                ids = ids[: self.max_length]
                label = label[:self.max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        return {'input_ids':input_ids, 'labels':labels}

class LoRATrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """只保存adapter"""
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

def train(global_args):
    hf_parser = HfArgumentParser(TrainingArguments)
    hf_train_args, = hf_parser.parse_json_file(json_file=global_args.train_args_json)

    set_seed(global_args.seed)
    hf_train_args.seed = global_args.seed
    model_max_length = global_args.max_input_length + global_args.max_output_length

    print("global_args.model_name_or_path", global_args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(global_args.model_name_or_path, trust_remote_code=True)
    lora_module_name = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama']

    config = LoraConfig(r=global_args.lora_dim,
                        lora_alpha=global_args.lora_alpha,
                        target_modules=lora_module_name,
                        lora_dropout=global_args.lora_dropout,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,)
    
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,device_map='auto').half().cuda()

    model = get_peft_model(model,config)
    resume_from_checkpoint = global_args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, 'adapter_model.bin'
            )
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            logger.info(f'Restarting from {checkpoint_name}')
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info(f"Checkpoint {checkpoint_name} not found")
        
    model.print_trainable_parameters()
    
    # data
    train_dataset = get_dataset(global_args.train_data_path, tokenizer, global_args)
    eval_dataset = None
    if global_args.eval_data_path:
        eval_dataset = get_dataset(global_args.eval_data_path, tokenizer, global_args)
    data_collator = DataCollatorForLlama2(pad_token_id=tokenizer.unk_token_id,max_length=model_max_length)

    print(train_dataset)

    # train
    trainer = LoRATrainer(
        model=model,
        args=hf_train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.model.save_pretrained(hf_train_args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)