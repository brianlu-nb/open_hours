In addition to the paramaters listed in `llama_finetuning_hf.py`, you can use all the parameters of Huggingface `Trainer`.
https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py

Multi-GPU training
```
torchrun --nnodes 1 --nproc_per_node 2 llama_finetuning_hf.py \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--output_dir exp/yelp_lora_test \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--dataset yelp_dataset \
--use_peft \
--peft_method lora \
--bf16 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--gradient_checkpointing
```

Single GPU training
```
python llama_finetuning_hf.py \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--output_dir exp/yelp_lora_test \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--dataset yelp_dataset \
--use_peft \
--peft_method lora \
--bf16 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--gradient_checkpointing
```