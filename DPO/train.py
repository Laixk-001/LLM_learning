from my_model import load_model,parse_args
from data_set import load_data
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig

"""
    DPO训练
"""

def train(train_data, dev_data, model, tokenizer, ref_model):
    """
    Args:
        train_data: 训练数据集
        dev_data: 验证数据集
        model: 模型
        tokenizer: 模型对应分词器
        ref_model: 参考模型
    
    Returns:
    """
    training_args = DPOConfig(
        per_device_train_batch_size=4, #单卡训练集batchsize
        per_device_eval_batch_size=4, #单卡验证集batchsize
        max_steps=200, #最大步长
        logging_steps=5, #日志打印步数
        save_steps=10, #模型保存步数
        gradient_accumulation_steps=4, #梯度累积步数
        gradient_checkpointing=True, #是否开启梯度保存
        learning_rate=5e-4, #学习率
        evaluation_strategy='steps', #验证策略
        eval_steps=20, #验证集经过多少步验证一次
        output_dir='/root/autodl-fs/dpo/result-dpo', #输出路径
        report_to='tensorboard', #报告显示在tensorboard
        lr_scheduler_type='cosine', #学习率调度器类型
        warmup_steps=100, #热身步数
        optim='adamw_hf', #优化器算法
        remove_unused_columns=False, #是否剔除数据集中不相关列
        bf16=False, #是否开启bf16
        fp16=True, #是否开启fp16
        run_name=f'dpo', #运行名称
    )

    #设置peft

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, #lora任务类型
        inference_mode=False, #是否开启推理模式
        r=8, #lora的秩
        lora_alpha=16, #归一化超参数
        lora_dropout=0.05, #dropout比例
    )

    trainer = DPOTrainer(
        model, #模型
        ref_model=ref_model, #参考模型
        force_use_ref_model=True, #是否使用参考模型
        args=training_args, #训练参数
        beta=0.1, #温度
        train_dataset=train_data, #训练集
        eval_dataset=dev_data, #验证集
        tokenizer=tokenizer, #分词器
        peft_config=peft_config, #peft配置参数
        max_prompt_length=256, #最大提示词长度
        max_length=512, #最大长度
    )
    train_result = trainer.train()
    trainer.save_model('/root/autodl-fs/dpo_result')

def main():
    arg_model = parse_args()
    model, tokenizer, ref_model = load_model(arg_model)
    train_data, dev_data = load_data()
    train(train_data,dev_data,model,tokenizer,ref_model)

if __name__ == '__main__':
    main()