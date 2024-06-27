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


template = (
    "We are building a RAG QA system in Newsbreak APP. Help me to process a collected real user query(conversation history is also provided).\n"
    "First determine whether the query is about a specific location or not(if so, we will add location constrains in the following retrieval process). "
    "Then rewrite the query to potentially improve the search result if needed. Rewrite only if:\n"
    "1. conversation history is not empty. Then replace all references in the query with specific information from the history to make the query self-contained.\n"
    "2. it is likely to be a local query but without a specific location mentioned. Then you may add the user location to the query.\n"
    "When rewriting, try not to alter the original query too much, and the rewritten query should not be too long.\n"
    "Then classify the query into at most two(should be only one for most cases) of the provided intents. You should also take the previous conversions into account if provided. Different intents will use different search/API to answer.\n"
    "Finally, extract slots.\n"
    "Conversation History:\n"
    "{history}\n"
    "User Query: {query}\n"
    "User Location: {location}\n\n"
    "Please provide your result as the JSON format below:\n"
    "{{\"is_local\": true or false, \"rewritten query\": \"self-contained query\", \"intents\": [\"intent1\", \"intent2\"], \"slots\": {{\"key\": \"value\"}}}}"
)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def process_dialog(dialog, tokenizer, min_turn_idx=0, return_prompt=False, train=False, train_on_context=-1):
    IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
    assert len(dialog)>=2
    dialog = dialog[:2*len(dialog)//2]
    inputs = []
    labels = []
    total_turns = len(dialog)//2
    prompt = ""
    for turn_idx in range(total_turns):
        cur_turn_text = f"{B_INST} {dialog[turn_idx*2].strip()} {E_INST} {dialog[turn_idx*2+1].strip()} "
        
        turn_input = [tokenizer.bos_token_id]+ \
                     tokenizer.encode(cur_turn_text, 
                                      add_special_tokens=False,
                                      truncation=False)+ \
                     [tokenizer.eos_token_id]
        if turn_idx>=min_turn_idx:
            cur_turn_only_input_text = f"{B_INST} {dialog[turn_idx*2].strip()} {E_INST}"
            turn_only_input = tokenizer.encode(cur_turn_only_input_text, 
                                            add_special_tokens=False,
                                            truncation=False)
            turn_label = turn_input.copy()
            input_len = len(turn_only_input)+1
            for i in range(input_len): # plus one for bos
                if train and random.random()<train_on_context and i>input_len*0.15 and i<input_len*0.85:
                    # this will include some token in the prompt for loss
                    continue
                turn_label[i] = IGNORE_INDEX
            prompt += cur_turn_only_input_text
        else:
            # for single turn training, we need to mask all history
            turn_label = [IGNORE_INDEX]*len(turn_input)
            prompt += cur_turn_text
        inputs.extend(turn_input)
        labels.extend(turn_label)
    if return_prompt:
        return prompt
    assert len(inputs)==len(labels)
    inputs = inputs[:tokenizer.model_max_length]
    labels = labels[:tokenizer.model_max_length]
    return inputs, labels

def format_location(user_loc):
    if user_loc is None:
        return 'Unknown'
    try:
        city, state = user_loc.split(',')
        new_loc = f"{city.strip().replace('_', ' ').capitalize()}, {state.strip().replace('_', ' ').capitalize()}"        
    except:
        new_loc = user_loc.strip().replace('_', ' ')
    return new_loc

def process_dialog_to_single_turn(data, tokenizer, return_prompt=False, train=False, train_on_context=-1, shuffle_ref=False):
    if 'date' in data and data['date'] is not None:
        date = data['date']
    else:
        date = datetime.now().strftime("%m-%d-%Y")
    prompt = template.format(
            history="\n".join(data['conversation']), 
            query=data['query'], 
            location=format_location(data['user_location'])
            )
    return process_dialog([prompt, json.dumps(data['label'], ensure_ascii=False)], tokenizer, min_turn_idx=0, return_prompt=return_prompt,
                          train=train,
                          train_on_context=train_on_context)


class QUDataset(Dataset):
    def __init__(self, tokenizer, config, train=False):
        self.ann = []
        if train:
            file = config.train_file
        else:
            file = config.eval_file
        with open(file, 'r') as f:
            for line in f:
                self.ann.append(json.loads(line))
        
        self.train = train
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]        
        inputs, labels = process_dialog_to_single_turn(ann, self.tokenizer, train=self.train)
        return {
            "input_ids": inputs,
            "labels": labels
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
    class qu_dataset:
        dataset: str = 'unified_dataset'
        train_file: str = "../data/qu_train_0327.jsonl"
    ds_config = qu_dataset()
    with open('../data/qu_train_0327.jsonl', 'r') as f:
        for line in f:
            d = json.loads(line)
            print(process_dialog_to_single_turn(d, tokenizer, return_prompt=True))
            break