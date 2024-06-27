export PYTHONPATH=$PYTHONPATH:./
export WANDB_API_KEY=9e19fcbe04cad945ebf6a0a5601293c89f8096cb

deepspeed --include=localhost:2,3,4,5 --master_port 29601 llama_dpo_hf.py \
--model_name_or_path ./exp/llama_13bf_1207 \
--output_dir exp/llama_13bf_1217_dpo \
--do_train \
--num_train_epochs 1 \
--learning_rate 2e-6 \
--beta 0.2 \
--dataset unified_dpo_dataset \
--bf16 True \
--tf32 True \
--use_flashatt_2 True \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--model_max_length 4096 \
--gradient_checkpointing True \
--report_to wandb \
--ddp_find_unused_parameters False \
--logging_steps 1 \
--run_name llama_13b_1217_dpo_16 \
--lr_scheduler 'cosine' \
--warmup_ratio 0.02 \
--save_steps 10000 \
--save_total_limit 2 \
--overwrite_output_dir \
--evaluation_strategy steps \
--eval_steps 80 \
--deepspeed ./configs/ds_3_hf_offload.json

# --run_name search_agent_11_02_2023_Instruct \
#--use_flashatt_2 True \
#--neftune_noise_alpha 15 \

