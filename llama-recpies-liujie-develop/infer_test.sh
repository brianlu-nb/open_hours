#!/bin/bash

#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tgi' --parallel_n 1
#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tgi' --parallel_n 2
#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tgi' --parallel_n 4
#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tgi' --parallel_n 8
#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tgi' --parallel_n 16
#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tgi' --parallel_n 32

#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tensorrt' --parallel_n 1
#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tensorrt' --parallel_n 2
#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tensorrt' --parallel_n 4
#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tensorrt' --parallel_n 8
#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tensorrt' --parallel_n 16
#python3 llama_infer.py --job 'infer_speed_multi_process' --model 'tensorrt' --parallel_n 32


python3 llama_infer.py --job 'infer_llama_multi_process' --model 'tgi' --parallel_n 8

