from . import unified_dataset
from . import content_quality_dataset
from . import content_genre_dataset
from torch.utils.data import Dataset
import json
import random


class MultiTaskDataset(Dataset):

    def __init__(self, tokenizer, config, train=False):
        self.data = []
        if train:
            file = config.train_file
        else:
            file = config.eval_file
        with open(file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        # note: can be done by building train data
        # random.shuffle(self.data)

        self.train = train
        self.tokenizer = tokenizer
        self.shuffle_ref = config.shuffle_ref

        self.content_quality_dataset = content_quality_dataset.ContentQualityDataset(None, None, init=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        one_data = self.data[index]
        if one_data["task_type"] == 'content_quality':
            # inputs, labels = self.content_quality_dataset.process_using_api(one_data, self.tokenizer)
            inputs, labels = self.content_quality_dataset.process_without_rationale(one_data, self.tokenizer)
        elif one_data["task_type"] == 'unified':
            inputs, labels = unified_dataset.process_dialog_to_single_turn(one_data, self.tokenizer, train=self.train, shuffle_ref=self.shuffle_ref)
        elif one_data["task_type"] == 'content_genre':
            # inputs, labels = content_genre_dataset.process_dialog_to_single_turn(one_data, self.tokenizer)
            inputs, labels = content_genre_dataset.process_without_reason(one_data, self.tokenizer)
        else:
            raise NotImplementedError(f"{one_data['task_type']} not implemented")

        return {
            "input_ids": inputs,
            "labels": labels
        }

