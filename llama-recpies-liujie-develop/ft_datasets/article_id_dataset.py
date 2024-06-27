from ft_datasets.unified_dataset import process_dialog
from torch.utils.data import Dataset
import json


def get_template():
    st_template = (
        "You are a helpful QA robot.\n"
        "Please recall the ID of the article which contains the following content:\n"
        "Content: {content}\n"
        "Your Reply:\n"
    )
    return st_template


def process_dialog_to_single_turn(data, tokenizer):
    prompt = get_template().format(content=data['label'])
    return process_dialog([prompt, data['docid']], tokenizer, min_turn_idx=0)



class ArticleIdDataset(Dataset):
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
        inputs, labels = process_dialog_to_single_turn(ann, self.tokenizer)
        return {
            "input_ids": inputs,
            "labels": labels
        }
    