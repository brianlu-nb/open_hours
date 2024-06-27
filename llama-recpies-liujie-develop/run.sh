# deepspeed --include=localhost:0,1,2,3 --master_port 29600 llama_finetuning_hf.py \
# --model_name_or_path meta-llama/Llama-2-13b-hf \
# --output_dir exp/llama_13bf_1207 \
# --do_train \
# --num_train_epochs 1 \
# --learning_rate 2e-5 \
# --dataset unified_dataset \
# --bf16 True \
# --tf32 True \
# --use_flashatt_2 True \
# --per_device_train_batch_size 8 \
# --per_device_eval_batch_size 8 \
# --gradient_accumulation_steps 2 \
# --model_max_length 4096 \
# --gradient_checkpointing True \
# --report_to wandb \
# --ddp_find_unused_parameters False \
# --logging_steps 1 \
# --run_name llama_13bf_1207 \
# --lr_scheduler 'cosine' \
# --warmup_ratio 0.02 \
# --save_steps 10000 \
# --save_total_limit 2 \
# --overwrite_output_dir \
# --evaluation_strategy steps \
# --eval_steps 80 \
# --deepspeed ./configs/ds_2_hf_offload.json
# --model_name_or_path /home/paperspace/xingguang/models/my_pre_train_dataset.v02.13b.1e-5.B6.E1.full/epoch_000.hf \

WANDB_API_KEY=1b6cab6d390168c39939c3a857b9a6711acfcd42 \
deepspeed --include localhost:0,1,2,3 --master_port 29600 llama_finetuning_hf.py \
--model_name_or_path meta-llama/Llama-2-13b-hf \
--output_dir exp/llama_13bf_0303 \
--do_train \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--dataset unified_dataset \
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
--run_name llama_13bf_0303 \
--lr_scheduler_type 'cosine' \
--warmup_ratio 0.02 \
--save_steps 10000 \
--save_total_limit 2 \
--overwrite_output_dir \
--evaluation_strategy steps \
--eval_steps 50 \
--shuffle_ref \
--deepspeed ./configs/ds_2_hf_offload.json

# deepspeed --include=localhost:0,1,2,3 --master_port 29600 llama_finetuning_hf.py \
# --model_name_or_path meta-llama/Llama-2-13b-hf \
# --output_dir exp/llama_13bf_1203_2 \
# --do_train \
# --num_train_epochs 1 \
# --learning_rate 2e-5 \
# --dataset unified_dataset \
# --bf16 True \
# --tf32 True \
# --use_flashatt_2 True \
# --per_device_train_batch_size 8 \
# --per_device_eval_batch_size 8 \
# --gradient_accumulation_steps 1 \
# --model_max_length 4096 \
# --gradient_checkpointing True \
# --report_to wandb \
# --ddp_find_unused_parameters False \
# --logging_steps 1 \
# --run_name llama_13bf_1203_2 \
# --lr_scheduler 'cosine' \
# --warmup_ratio 0.02 \
# --save_steps 10000 \
# --save_total_limit 2 \
# --overwrite_output_dir \
# --evaluation_strategy steps \
# --eval_steps 80 \
# --deepspeed ./configs/ds_2_hf_offload.json