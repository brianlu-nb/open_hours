# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

# references:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py#L382
# https://github.com/facebookresearch/llama-recipes/blob/03faba661f079ee1ecaeb66deaa6bdec920a7bab/inference/chat_utils.py#L19

import json
import random
from datetime import datetime
import re
from torch.utils.data import Dataset
from ft_datasets.unified_dataset import process_dialog_to_single_turn
# each line of the input file is one training/validation/test sample in JSON format
# The format of each sample is as follows:
# {
#     "refs": [],
#     "messages": [
#         "Hi",
#         "Hello, what can I do for you?"
#     ],
#     "multiturn_prob": 0.3
# }
# the refs is optional
# messages should contain 2*n of text. arranged alternately in the order of input and response



class UnifiedDPODataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name='train'):
        self.ann = []
        with open(split_name, 'r') as f:
            for line in f:
                self.ann.append(json.loads(line))
        if split_name.find('train')>=0:
            self.train = True
        else:
            self.train = False
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]        
        # if self.train and ann['source']!='platypus' and random.random()<0.1:
        #     another_idx = random.randint(0, len(self.ann)-1)
        #     if another_idx!=index:
        #         ann = dict(ann)
        #         ann['history'] = [self.ann[another_idx]['query'], self.ann[another_idx]['answer']]+ann['history']
        prompt = process_dialog_to_single_turn(ann, self.tokenizer, return_prompt=True)
        return {
            "prompt": prompt,
            "chosen": ann['answer'].strip(),
            "rejected": ann['rejected_response'].strip()
        }
    

if __name__ == '__main__':
    from transformers import AutoTokenizer, AddedToken
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    print(tokenizer.model_max_length)
    tokenizer.model_max_length=4096
    print(tokenizer.model_max_length)
    # print(tokenizer.eos_token)
    # print(tokenizer.bos_token)
    # print(tokenizer.eos_token_id)
    # print(tokenizer.bos_token_id)
    print(tokenizer.tokenize(f"{B_INST} {E_INST} a a<cite> 1 </cite>"))
    print(tokenizer.tokenize(f"{B_INST} {E_INST}"))
    tokenizer.add_tokens(["<cite>", "</cite>"])
    print(tokenizer.tokenize(f"{B_INST} {E_INST} a a <cite> 1 </cite><cite> 2 </cite>"))
    # print(tokenizer.encode(f"{B_INST} hi {E_INST}"))
    # print(process_dialog([
    #     "a a a a",
    #     "b b b b",
    #     "a a a a",
    #     "b b b b"
    # ], tokenizer))
    from dataclasses import dataclass
    @dataclass
    class unified_dataset:
        dataset: str = 'unified_dataset'
        train_split: str = "../data/unified_train_0914.jsonl"
        test_split: str = "ft_datasets/unified_valid.jsonl"
    ds_config = unified_dataset()
    ds = UnifiedDataset(ds_config, tokenizer, split_name="../data/unified_train_0914.jsonl")
    # d = ds[16202]
    # print(d['labels'])
    # max_len = 0
    # count = 0
    # for item in iter(ds):
    #     if len(item['input_ids'])>max_len:
    #         max_len = len(item['input_ids'])
    #         print(max_len)
    #     label_num = sum([int(x!=-100) for x in item['labels']])
    #     if label_num<10:
    #         print(count)
    #         break
    #     print('label:', label_num)
    #     count += 1
    # print(max_len)
    # print(get_template(cite=True, multi_turn=False).format(history='history',background='refs', date='2023-09-06', question='question'))