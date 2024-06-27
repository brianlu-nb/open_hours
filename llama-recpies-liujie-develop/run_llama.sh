#!/bin/bash

llama3_content_quality="llama3_content_quality"
llama3_multitask="llama3_multitask"
llama2_content_quality="llama2_content_quality"
llama2_multitask="llama2_multitask"


job="${llama3_multitask}"

if [ "${job}" == "llama3_content_quality" ]; then
  echo "run job llama3_content_quality"
  WANDB_API_KEY=1b6cab6d390168c39939c3a857b9a6711acfcd42 \
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_node 6 --master_port 29600 llama_finetuning_hf.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --output_dir exp/llama3_v4 \
  --do_train \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --dataset content_quality_dataset \
  --train_file /mnt/share16t/liujie/code_workspace/dataset/content_quality_dataset/train.jsonl \
  --eval_file /mnt/share16t/liujie/code_workspace/dataset/content_quality_dataset/test.jsonl \
  --bf16 True \
  --tf32 True \
  --use_flashatt_2 True \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --model_max_length 8192 \
  --report_to wandb \
  --ddp_find_unused_parameters False \
  --logging_steps 1 \
  --run_name llama3_v4 \
  --lr_scheduler_type 'cosine' \
  --warmup_ratio 0.02 \
  --save_steps 10000 \
  --save_total_limit 1 \
  --overwrite_output_dir \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --shuffle_ref \
  --fsdp "shard_grad_op offload auto_wrap" \
  --fsdp_config ./configs/fsdp_config.json
elif [ "${job}" == "llama3_multitask" ]; then
  echo "run job llama3_open_hours"
  WANDB_API_KEY=c93344ef5b6572250edee53ca8ee0a765ce1b48a \
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_node 6 --master_port 29600 llama_finetuning_hf.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --output_dir exp/llama3_multitask_v6 \
  --do_train \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --dataset open_hours_dataset \
  --train_file /root/brianlu/test_hours/train.jsonl \
  --eval_file /root/brianlu/test_hours/test.jsonl \
  --bf16 True \
  --tf32 True \
  --use_flashatt_2 True \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --model_max_length 8192 \
  --report_to wandb \
  --ddp_find_unused_parameters False \
  --logging_steps 1 \
  --run_name llama3_multitask_v6 \
  --lr_scheduler_type 'cosine' \
  --warmup_ratio 0.02 \
  --save_steps 10000 \
  --save_total_limit 1 \
  --overwrite_output_dir \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --shuffle_ref \
  --deepspeed ./configs/ds_2_hf_offload.json
elif [ "${job}" == "llama2_content_quality" ]; then
  echo "run job llama2_content_quality"
  WANDB_API_KEY=1b6cab6d390168c39939c3a857b9a6711acfcd42 \
  deepspeed --include localhost:5,7 --master_port 29600 llama_finetuning_hf.py \
  --model_name_or_path meta-llama/Llama-2-13b-hf \
  --output_dir exp/llama_13bf_finetune_v4 \
  --do_train \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --dataset content_quality_dataset \
  --bf16 True \
  --tf32 True \
  --use_flashatt_2 True \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --report_to wandb \
  --ddp_find_unused_parameters False \
  --logging_steps 1 \
  --run_name llama_13bf_finetune_v4 \
  --lr_scheduler_type 'cosine' \
  --warmup_ratio 0.02 \
  --save_steps 10000 \
  --save_total_limit 2 \
  --overwrite_output_dir \
  --evaluation_strategy steps \
  --eval_steps 10 \
  --shuffle_ref \
  --deepspeed ./configs/ds_2_hf_offload.json
elif [ "${job}" == "llama2_multitask" ]; then
  echo "run job llama2_multitask"
  WANDB_API_KEY=1b6cab6d390168c39939c3a857b9a6711acfcd42 \
  deepspeed --include localhost:1,7 --master_port 29600 llama_finetuning_hf.py \
  --model_name_or_path meta-llama/Llama-2-13b-hf \
  --output_dir exp/llama_13bf_multitask_v2 \
  --do_train \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --dataset multitask \
  --train_file /mnt/share16t/liujie/code_workspace/dataset/multitask_dataset/multitask_train_0411.jsonl \
  --eval_file /mnt/share16t/liujie/code_workspace/dataset/multitask_dataset/multitask_test_0411.jsonl \
  --bf16 True \
  --tf32 True \
  --use_flashatt_2 True \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --report_to wandb \
  --ddp_find_unused_parameters False \
  --logging_steps 1 \
  --run_name llama_13bf_multitask_v2 \
  --lr_scheduler_type 'cosine' \
  --warmup_ratio 0.02 \
  --save_steps 10000 \
  --save_total_limit 1 \
  --overwrite_output_dir \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --shuffle_ref \
  --deepspeed ./configs/ds_2_hf_offload.json
else
  echo "job is not supported error, please check your job name!"
  exit 1
fi
