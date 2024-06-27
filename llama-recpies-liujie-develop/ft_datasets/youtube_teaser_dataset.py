from ft_datasets.unified_dataset import process_dialog
from torch.utils.data import Dataset
import json


def get_template():
    st_template = (
        "You are a helpful QA robot.\n"
        "Below is a video's title, description, and transcript. Please write a news article based on the input video:\n"
        "Video title: {video_title}\n"
        "Video description: {video_description}\n"
        "Video transcript: {video_transcript}\n"
        "Your Reply:\n"
    )
    return st_template


def process_dialog_to_single_turn(data, tokenizer):
    prompt = get_template().format(video_title=data['video_title'], 
                                   video_description=data['video_description'], 
                                   video_transcript=data['video_transcript'])
    output = f"Article title: {data['aigc_title']}\nArticle content: {data['aigc_article']}"
    return process_dialog([prompt, output], tokenizer, min_turn_idx=0)


def data_to_prompt(data):
    prompt = get_template().format(video_title=data['video_title'], 
                                   video_description=data['video_description'], 
                                   video_transcript=" ".join(data['video_transcript'].split(' ')[0:500]))
    return f"[INST] {prompt} [/INST]"


class YoutubeTeaserDataset(Dataset):
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
    