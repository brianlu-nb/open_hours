from torch.utils.data import Dataset
from datetime import datetime
import json
import random
import sys
import re

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
S_HEADER, E_HEADER = "<|start_header_id|>", "<|end_header_id|>"
EOT = "<|eot_id|>"


def process_dialog_to_single_turn(data, tokenizer):
    prompt = prompt_templates.to_llama_format(data)
    output = json.dumps(data["expected_response"], indent=4)
    return process_dialog([prompt, output], tokenizer, min_turn_idx=0)


def process_dialog(dialog, tokenizer, min_turn_idx=0, return_prompt=False, train=False, train_on_context=-1):
    IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
    assert len(dialog)>=2
    dialog = dialog[:2*len(dialog)//2]
    inputs = []
    labels = []
    total_turns = len(dialog)//2
    prompt = ""
    for turn_idx in range(total_turns):
        cur_turn_text = f"{dialog[turn_idx*2]}{dialog[turn_idx*2+1].strip()} "
        
        turn_input = [tokenizer.bos_token_id]+ \
                     tokenizer.encode(cur_turn_text, 
                                      add_special_tokens=False,
                                      truncation=False)+ \
                     [tokenizer.eos_token_id]
        if turn_idx>=min_turn_idx:
            cur_turn_only_input_text = dialog[turn_idx*2]
            turn_only_input = tokenizer.encode(cur_turn_only_input_text, 
                                            add_special_tokens=False,
                                            truncation=False)
            turn_label = turn_input.copy()
            input_len = len(turn_only_input)
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

class OpenHoursDataset(Dataset):
    def __init__(self, tokenizer, config, train=False):
        self.ann = []
        if train:
            file = config.train_file
        else:
            file = config.eval_file
        with open(file, 'r') as f:
            for line in f:
                self.ann.append(json.loads(line))
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
        
if __name__ == "__main__":
    from transformers import AutoTokenizer
    import open_hours.prompt_templates
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    from dataclasses import dataclass
    @dataclass
    class open_hours_dataset:
        dataset: str = 'open_hours_dataset'
        train_file: str = "open_hours/data/Current_Run_2024_07_26_18_37_24/train.jsonl"
        eval_file: str = "open_hours/data/Current_Run_2024_07_26_18_37_24/test.jsonl"
    
    ds = OpenHoursDataset(tokenizer=tokenizer, config=open_hours_dataset())
    input = ds[0]['input_ids']
    print(ds[0])
    print(tokenizer.decode(input))
    