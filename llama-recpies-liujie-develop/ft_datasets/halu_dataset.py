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

sample = {
    "baseless info": [
      "span1",
      "span2"
    ],
    "conflict": [
      "span1",
      "span2"
    ]
  }

INSTRUCTION = (
    "Please review the content and determine whether it contains either or both of the following two types of hallucinations and list the hallucination spans:\n" 
    "1. conflict: instances where the generative content presents direct contraction or opposition to the original input;\n"
    "2. baseless info: instances where the generated content includes information which is not substantiated by or inferred from the original input. However, logical and sensible inferences rooted in the common sense or general advice or sentence without actual meaning, such as courtesy phrases, should not be considered as baseless info, even if they are not explicitly mentioned in the original input.\n"
    "Please output the result in the following JSON format:\n"
    f"{json.dumps(sample, indent=2)}\n"
    "Your output:"
)


TEMPLATES = {
    "QA": (
    "Given a question:\n"
    "{question}\n\n"
    "some references: \"\"\"\n{reference}\n\"\"\"\n\n"
    "and an answer to check:\n{response}\n\n"
    # f"{INSTRUCTION}"
    ),
    "Summary": (
        "Given an article:\n"
        "\"\"\"\n{reference}\n\"\"\"\n\n"
        "and a summary of the article to check:\n{response}\n\n"
        # f"{INSTRUCTION}"
    ),
    "Data2txt": (
        "Given an JSON data about a local business:\n"
        "\"\"\"\n{reference}\n\"\"\"\n\n"
        "and an overview of the business based on the JSON data to check:\n{response}\n\n"
        # f"{INSTRUCTION}"
    ),
    "Youtube": (
        "Given an article:\n"
        "\"\"\"\n{reference}\n\"\"\"\n\n"
        "and a summary of the article to check:\n{response}\n\n"
    )
}

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

def process_dialog_to_single_turn(data, tokenizer, return_prompt=False):
    if data['task_type']=='QA':
        prompt = TEMPLATES[data['task_type']].format(
            question=data['question'], 
            reference=data['reference'], 
            response=data['response']
        )
    else:
        prompt = TEMPLATES[data['task_type']].format(
            reference=data['reference'], 
            response=data['response']
        )

    prompt += INSTRUCTION
        
    if return_prompt:
        return prompt

    return process_dialog([prompt, json.dumps(data['format_label'], indent=2)], tokenizer)
    

class DetectDataset(Dataset):
    def __init__(self, tokenizer, config, train=False):
        self.ann = []
        if train:
            file = config.train_file
        else:
            file = config.eval_file
        with open(file, 'r') as f:
            for line in f:
                d = json.loads(line)
                self.ann.append(d)
        
        self.train = train
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]        
        inputs, labels = process_dialog_to_single_turn(ann, self.tokenizer)
        return {
            "input_ids": inputs,
            "labels": labels
        }
    

