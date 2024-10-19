python3 train_lora_llama.py --train_args_json ./llama2-7B_LoRA.json  \
                            --train_data_path ../data/train.jsonl  \
                            --eval_data_path ../data/dev.jsonl  \
                            --model_name_or_path /root/autodl-fs/Llama2-Chinese-7b  \
                            --seed 42  \
                            --max_input_length 1024  \
                            --max_output_length 1024  \
                            --lora_rank 4  \
                            --lora_dim 8