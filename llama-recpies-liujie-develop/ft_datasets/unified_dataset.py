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

cite_pattern1 = re.compile(r"<cite>[0-9\,\s]+</cite>")
cite_pattern2 = re.compile(r"\[[0-9\,\s]+\]")
cot_pattern = re.compile(r"<cot>.*<\/cot>")

def clean_citation_and_cot(text):
    ret = re.sub(cite_pattern1, "", text)
    ret = re.sub(cite_pattern2, "", ret)
    ret = re.sub(cot_pattern, "", ret)
    return ret.strip()

default_rules = (
    "1. Use only the information you've discovered to answer the question. Do not make up information.\n"
    "2. Pay special attention to the information such as time, location, and people involved. Do not make any mistakes.\n"
    "3. If the answer isn't in the found information, inform the user and suggest related questions that the data can answer.\n"
    "4. Refer to the data as \"information I found\" rather than \"information provided\".\n"
    "5. For broad queries, summarize the most pertinent information in bullet points, prefacing your answer with a general overview.\n"
    "6. Prioritize referencing articles from high-quality sources, but still refer to articles with poor source quality if they contain relevant information.\n"
    "7. For event queries, if the user does not specify a time range, prioritize returning events that have not yet ended.\n"
)

def get_template():
    st_template = (
        "You are a helpful QA robot.\n"
        "Now is {date}.\n"
        "Given a question from a {user_location} user: {query}\n"
        "Answer the question based on the following information you find:\n"
        "{background}\n\n"
        "{rules}"
        "Your Reply:\n"
    )
    
    return st_template

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def make_background(refs, shuffle=False):
    if len(refs)==0:
        return "The search engine seems to be temporarily out of order and no relevant information was found."
    ret = ""
    ret = []
    for idx, r in enumerate(refs):
        ret.append(f"Article No.: {idx+1}\n{r}")
    if shuffle:
        random.shuffle(ret)
    ret = "\n----------------\n".join(ret)
    return ret

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

def process_dialog_to_single_turn(data, tokenizer, return_prompt=False, train=False, train_on_context=-1, shuffle_ref=False):
    if data['source']=='platypus':
        # do not train on context
        return process_dialog([data['query'], data['answer']], tokenizer)
    if data['source'] in ['no_robot', 'lima','news_new','judge']:
        return process_dialog(data['dialog'], tokenizer)
    refs = list(data['refs'])
    if train and shuffle_ref and random.random()<0.15:
        background = make_background(refs, shuffle=True)
    else:
        background = make_background(refs)

    if 'date' in data and data['date'] is not None:
        date = data['date']
    else:
        date = datetime.now().strftime("%m-%d-%Y")
    prompt = get_template().format(
            background=background, 
            query=data['query'], 
            date=date,
            rules=data.get('rules', default_rules),
            user_location=data.get('user_loc', 'America'))
    history = [clean_citation_and_cot(h) for h in data['history']]
    return process_dialog(history+[prompt, data['answer']], tokenizer, min_turn_idx=len(data['history'])//2, return_prompt=return_prompt,
                          train=train,
                          train_on_context=train_on_context)


class UnifiedDataset(Dataset):
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
        self.train_on_context = -1
        self.shuffle_ref = config.shuffle_ref


    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]        
        if self.train and ann['source'] not in ['platypus','no_robot','lima','news_new','judge'] and random.random()<0.1:
            another_idx = random.randint(0, len(self.ann)-1)
            if another_idx!=index and self.ann[another_idx]['source']=='real_search':
                ann = dict(ann)
                ann['history'] = [self.ann[another_idx]['query'], self.ann[another_idx]['answer']]+ann['history']
        inputs, labels = process_dialog_to_single_turn(ann, self.tokenizer, train=self.train, train_on_context=self.train_on_context,
                                                       shuffle_ref=self.shuffle_ref)
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