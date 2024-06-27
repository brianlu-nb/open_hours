# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

# references:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py#L382
# https://github.com/facebookresearch/llama-recipes/blob/03faba661f079ee1ecaeb66deaa6bdec920a7bab/inference/chat_utils.py#L19

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

# each line of the input file is one training/validation/test sample in JSON format
# The format of each sample is as follows:
# {
#     "system": "the system prompt, can be None",
#     "messages": [
#         "Hi",
#         "Hello, what can I do for you?"
#     ]
# }
# the system is optional
# messages should contain 2*n of text. arranged alternately in the order of input and response

SYSTEM_TEMPLATE = (
    "Below is some background information: "
    "{system}"
    "Answer questinos based on the given info, and provide the reference No. in the end of you answer."
    "If the question cannot be answered with the information given, please output a search query beginning with 'query:'"
)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def process_dialog(dialog, tokenizer, only_first_turn=False):
    IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
    assert len(dialog)%2==0
    inputs = []
    labels = []
    total_turns = len(dialog)//2
    for turn_idx in range(total_turns):
        turn_input = [tokenizer.bos_token_id]+ \
                     tokenizer.encode(f"{dialog[turn_idx*2].strip()} {dialog[turn_idx*2+1].strip()}", add_special_tokens=False,
                                      truncation=True, 
                                      max_length=tokenizer.model_max_length-2)+ \
                     [tokenizer.eos_token_id]
        turn_only_input = tokenizer.encode(f"{dialog[turn_idx*2].strip()}", 
                                           add_special_tokens=False,
                                           truncation=True, 
                                           max_length=tokenizer.model_max_length-1)
        turn_label = turn_input.copy()
        for i in range(len(turn_only_input)+1): # plus one for bos
            turn_label[i] = IGNORE_INDEX
        inputs.extend(turn_input)
        labels.extend(turn_label)
        if only_first_turn:
            break
    assert len(inputs)==len(labels)
    return inputs, labels

class MultiTurnDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name='train'):
        self.ann = []
        # if split_name=='train':
        #     file_path = dataset_config.train_split
        # else:
        #     file_path = dataset_config.test_split
        with open(split_name, 'r') as f:
            for line in f:
                self.ann.append(json.loads(line))

        self.tokenizer = tokenizer
        self.system_template = dataset_config.system_template
        self.use_system_prompt = dataset_config.use_system_prompt
        self.only_first_turn = dataset_config.only_first_turn

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        # process system prompt
        if ann.get('system') is not None:
            if self.system_template is not None:
                system = self.system_template.format(system=ann['system'])
            else:
                system = ann['system']
            dialog = [ 
                        system + "\n" + ann['messages'][0]
                    ] + ann['messages'][1:]
        else:
            dialog = ann['messages']
        
        inputs, labels = process_dialog(dialog, self.tokenizer, self.only_first_turn)
        
        return {
            "input_ids": inputs,
            "labels": labels
        }

if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    print(tokenizer.model_max_length)
    tokenizer.model_max_length=10000
    print(tokenizer.model_max_length)
    # print(tokenizer.eos_token)
    # print(tokenizer.bos_token)
    # print(tokenizer.eos_token_id)
    # print(tokenizer.bos_token_id)
    print(tokenizer.tokenize(f"<s> {B_INST}"))
    print(tokenizer.encode(f"{B_INST}"))
    # print(tokenizer.encode(f"{B_INST} hi {E_INST}"))
    print(process_dialog([
        "a a a a",
        "b b b b",
        "a a a a",
        "b b b b"
    ], tokenizer, only_first_turn=True))
    # from dataclasses import dataclass
    # @dataclass
    # class yelp_dataset:
    #     dataset: str = 'yelp_dataset'
    #     train_split: str = "ft_datasets/yp_train.jsonl"
    #     test_split: str = "ft_datasets/yp_valid.jsonl"
    #     system_template = None
    #     use_system_prompt = False
    #     only_first_turn = False
    # ds_config = yelp_dataset()
    # max_len = 0
    # ds = MultiTurnDataset(ds_config, tokenizer, split_name='./yp_train.jsonl')
    # for item in iter(ds):
    #     if len(item['input_ids'])>max_len:
    #         max_len = len(item['input_ids'])
    # print(max_len)