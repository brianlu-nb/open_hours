set -x #echo on
set -e #stop when any command fails


export PYTHONPATH=$PYTHONPATH:/home/paperspace/chengniu/llama-recpies
export WANDB_API_KEY=e3d908763df7ae980a3cf4f8398393dfcde15b64
export WANDB_PROJECT=youtube_teaser

deepspeed --include=localhost:0,1 --master_port=53000 llama_finetuning_hf.py \
--model_name_or_path meta-llama/Llama-2-13b-hf \
--output_dir /mnt/share_new/chengniu/exp/13b_youtube_teaser \
--do_train \
--num_train_epochs 2 \
--learning_rate 2e-5 \
--dataset youtube_teaser_dataset \
--bf16 True \
--tf32 True \
--use_fast_kernels True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--model_max_length 4096 \
--gradient_checkpointing True \
--report_to wandb \
--ddp_find_unused_parameters False \
--logging_steps 1 \
--run_name llama_13b_youtube_teaser \
--lr_scheduler 'cosine' \
--warmup_ratio 0.02 \
--save_steps 1000 \
--save_total_limit 2 \
--overwrite_output_dir \
--evaluation_strategy steps \
--eval_steps 1000 \
--deepspeed ./configs/ds_2_hf_offload.json \
