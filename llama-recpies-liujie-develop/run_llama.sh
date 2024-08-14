#!/bin/bash

llama3_content_quality="llama3_content_quality"
llama3_multitask="llama3_multitask"
llama2_content_quality="llama2_content_quality"
llama2_multitask="llama2_multitask"
llama3_open_hours="llama3_open_hours"


job="${llama3_open_hours}"

if [ "${job}" == "llama3_open_hours" ]; then
  echo "run job llama3_open_hours"
  WANDB_API_KEY=c93344ef5b6572250edee53ca8ee0a765ce1b48a \
  CUDA_VISIBLE_DEVICES=2,3,6,7 torchrun --nnodes 1 --nproc_per_node 4 --master_port 29600 llama_finetuning_hf.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --output_dir exp/llama3_open_hours \
  --do_train \
  --num_train_epochs 1 \
  --learning_rate 1e-6 \
  --dataset open_hours_dataset \
  --train_file /root/brianlu/open_hours/data/Current_Run_2024_07_26_18_37_24/train.jsonl \
  --eval_file /root/brianlu/open_hours/data/Current_Run_2024_07_26_18_37_24/test.jsonl \
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
  --run_name llama3_open_hours \
  --lr_scheduler_type 'cosine' \
  --warmup_ratio 0.02 \
  --save_steps 10000 \
  --save_total_limit 1 \
  --overwrite_output_dir \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --shuffle_ref \
  --deepspeed ./configs/ds_2_hf_offload.json
else
  echo "job is not supported error, please check your job name!"
  exit 1
fi
