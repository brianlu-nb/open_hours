from torch.utils.data import Dataset
from datetime import datetime
import json
import random
import sys
import re

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
S_HEADER, E_HEADER = "<|start_header_id|>", "<|end_header_id|>"
EOT = "<|eot_id|>"

GIVE_REASON = True

#-----------------------------------^^^---WAS DONE---^^^-----------------------------------#

# st_template = (
#     "You are a helpful QA robot.\n"
#     "Now is {date}.\n"
#     "Given a question from a {user_location} user: {query}\n"
#     "Answer the question based on the following information you find:\n"
#     "{background}\n\n"
#     "{rules}"
#     "Your Reply:\n"
# )
#-----------------------------------VVV---WAS DONE---VVV-----------------------------------#
def process_dialog_to_single_turn(data, tokenizer):
    opening_hours = data["opening_hours"]
    today = data["today"]
    user_prompt = data["user_prompt"]

    if GIVE_REASON:
        path = '/root/brianlu/test_hours/prompt.txt'
    else:
        path = '/root/brianlu/test_hours/prompt_without_reason.txt'
        if "reasoning" in data["expected_response"]:
            del data["expected_response"]["reasoning"]

    with open(path, 'r') as file:
        prompt = file.read().format(opening_hours=json.dumps(opening_hours, indent=4), today=today, user_prompt=user_prompt)
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
#-----------------------------------^^^---WAS DONE---^^^-----------------------------------#

# def make_background(refs, shuffle=False):
#     if len(refs)==0:
#         return "The search engine seems to be temporarily out of order and no relevant information was found."
#     ret = ""
#     ret = []
#     for idx, r in enumerate(refs):
#         ret.append(f"Article No.: {idx+1}\n{r}")
#     if shuffle:
#         random.shuffle(ret)
#     ret = "\n----------------\n".join(ret)
#     return ret


# default_rules = (
#     "1. Use only the information you've discovered to answer the question. Do not make up information.\n"
#     "2. Pay special attention to the information such as time, location, and people involved. Do not make any mistakes.\n"
#     "3. If the answer isn't in the found information, inform the user and suggest related questions that the data can answer.\n"
#     "4. Refer to the data as \"information I found\" rather than \"information provided\".\n"
#     "5. For broad queries, summarize the most pertinent information in bullet points, prefacing your answer with a general overview.\n"
#     "6. Prioritize referencing articles from high-quality sources, but still refer to articles with poor source quality if they contain relevant information.\n"
#     "7. For event queries, if the user does not specify a time range, prioritize returning events that have not yet ended.\n"
# )
# cite_pattern1 = re.compile(r"<cite>[0-9\,\s]+</cite>")
# cite_pattern2 = re.compile(r"\[[0-9\,\s]+\]")
# cot_pattern = re.compile(r"<cot>.*<\/cot>")

# def clean_citation_and_cot(text):
#     ret = re.sub(cite_pattern1, "", text)
#     ret = re.sub(cite_pattern2, "", ret)
#     ret = re.sub(cot_pattern, "", ret)
#     return ret.strip()

# def process_dialog_to_single_turn(data, tokenizer, return_prompt=False, train=False, train_on_context=-1, shuffle_ref=False):
#     refs = list(data['refs'])
#     if train and shuffle_ref and random.random()<0.15:
#         background = make_background(refs, shuffle=True)
#     else:
#         background = make_background(refs)

#     if 'date' in data and data['date'] is not None:
#         date = data['date']
#     else:
#         date = datetime.now().strftime("%m-%d-%Y")
#     prompt = st_template.format(
#             background=background, 
#             query=data['query'], 
#             date=date,
#             rules=data.get('rules', default_rules),
#             user_location=data.get('user_loc', 'America'))
#     history = [clean_citation_and_cot(h) for h in data['history']]
#     return process_dialog(history+[prompt, data['answer']], tokenizer, min_turn_idx=len(data['history'])//2, return_prompt=return_prompt,
#                           train=train,
#                           train_on_context=train_on_context)

# def process_without_reason(data, tokenizer):
#     title = data["title"]
#     content = data["content"]

#     prompt = finetune_prompt.content_genre_without_reason_template.format(title=title, content=content)
#     gpt_result = prompt_util.remove_content_genre_reason(data["gpt_result"])
#     output = json.dumps(gpt_result)
#     return process_dialog([prompt, output], tokenizer, min_turn_idx=0)


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
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    from dataclasses import dataclass
    @dataclass
    class open_hours_dataset:
        dataset: str = 'open_hours_dataset'
        train_file: str = "test_hours/data.jsonl"
        eval_file: str = "test_hours/data.jsonl"
    
    ds = OpenHoursDataset(tokenizer=tokenizer, config=open_hours_dataset())
    input = ds[0]['input_ids']
    print(ds[0])
    print(tokenizer.decode(input))
    