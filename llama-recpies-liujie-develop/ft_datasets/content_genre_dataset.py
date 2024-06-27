from . import unified_dataset, prompt_util
from . import finetune_prompt
from torch.utils.data import Dataset
import json
import sys


def process_dialog_to_single_turn(data, tokenizer):
    title = data["title"]
    content = data["content"]
    title = prompt_util.get_by_limit_len_title(title)
    content = prompt_util.get_by_limit_len_content(content)

    prompt = finetune_prompt.content_genre_template.format(title=title, content=content)
    output = json.dumps(data["gpt_result"])
    return unified_dataset.process_dialog([prompt, output], tokenizer, min_turn_idx=0)


def process_without_reason(data, tokenizer):
    title = data["title"]
    content = data["content"]
    title = prompt_util.get_by_limit_len_title(title)
    content = prompt_util.get_by_limit_len_content(content)

    prompt = finetune_prompt.content_genre_without_reason_template.format(title=title, content=content)
    gpt_result = prompt_util.remove_content_genre_reason(data["gpt_result"])
    output = json.dumps(gpt_result)
    return unified_dataset.process_dialog([prompt, output], tokenizer, min_turn_idx=0)


class ContentGenreDataset(Dataset):
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
        inputs, labels = unified_dataset.process_dialog_to_single_turn(ann, self.tokenizer)
        return {
            "input_ids": inputs,
            "labels": labels
        }
