#!/bin/bash

#conda init
#conda activate py10
#source /home/paperspace/miniconda3/bin/activate py10
#
#nohup bash run_llama.sh > nohup.out 2>&1 &

#python3 llama_infer.py --job infer_speed_multi_process --model tgi --parallel_n 1
#python3 llama_infer.py --job infer_speed_multi_process --model tgi --parallel_n 2
#python3 llama_infer.py --job infer_speed_multi_process --model tgi --parallel_n 4
#python3 llama_infer.py --job infer_speed_multi_process --model tgi --parallel_n 8
#python3 llama_infer.py --job infer_speed_multi_process --model tgi --parallel_n 16
#python3 llama_infer.py --job infer_speed_multi_process --model tgi --parallel_n 32

python3 llama_infer.py --job infer_llama_multi_process --model tgi --parallel_n 384 \
 --input_file "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240401.low_quality.jsonl" \
 --output_file "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240401.low_quality.writing_quality.jsonl"

python3 llama_infer.py --job infer_llama_multi_process --model tgi --parallel_n 384 \
 --input_file "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240501.low_quality.jsonl" \
 --output_file "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240501.writing_quality.low_quality.jsonl"

python3 llama_infer.py --job infer_llama_multi_process --model tgi --parallel_n 384 \
 --input_file "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240516.low_quality.jsonl" \
 --output_file "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240516.writing_quality.low_quality.jsonl"

python3 llama_infer.py --job infer_llama_multi_process --model tgi --parallel_n 384 \
 --input_file "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240522.low_quality.jsonl" \
 --output_file "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240522.writing_quality.low_quality.jsonl"