from huggingface_hub import InferenceClient, AsyncInferenceClient
from argparse import ArgumentParser
from unified_dataset import process_dialog_to_single_turn
from diskcache import Cache
from transformers import AutoTokenizer
import json
import asyncio
import time
from tqdm import tqdm

# client = AsyncInferenceClient(model="http://10.128.0.54:8300", timeout=100)
client = AsyncInferenceClient(model="http://184.105.106.16:8300", timeout=100)
# actually we do not need tokenizer
# just to meet the parameter requirements
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')

async def generate_response(data, idx, cache):
    if idx in cache:
        return cache[idx]
    
    # get input prompt
    input_prompt = process_dialog_to_single_turn(data, tokenizer, return_prompt=True)
    max_token = 4000-len(tokenizer.encode(input_prompt))
    answer = await client.text_generation(input_prompt,
                                          max_new_tokens=max_token, 
                                          stream=False,
                                          do_sample=True,
                                          temperature=0.7,
                                          top_p=0.95,
                                          top_k=40)
    ret = dict(data)
    ret['rejected_response'] = answer
    cache[idx] = ret
    return ret

async def main(args):
    cache = Cache(f'../cache/{args.model_name}')
    idx = 0
    tasks = []
    final_data = []
    with open(args.raw_dataset, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            if data['source'] not in ['real_search', 'yelp']:
                idx += 1
                continue
            tasks.append(asyncio.create_task(generate_response(data, idx, cache)))
            idx += 1
            if len(tasks)==args.parallel:
                start = time.perf_counter()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                final_data.extend([x for x in results if isinstance(x, dict)])
                print(f'current idx: {idx}, gathered data: {len(final_data)}, time used: {time.perf_counter()-start:.3f}s')
                tasks = []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_data.extend([x for x in results if isinstance(x, dict)])
    with open(args.output_file, 'w') as f:
        for d in final_data:
            f.write(json.dumps(d)+"\n")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--raw_dataset', default="../data/unified_train_1207.jsonl")
    parser.add_argument('--output_file', default="../data/gy_dpo_1211.jsonl")
    parser.add_argument('--model_name', default='1211_gy_dpo')
    parser.add_argument('--parallel', type=int, default=5)
    
    args = parser.parse_args()
    asyncio.run(main(args))

    
