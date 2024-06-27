import copy
import json
import sys

from torch.utils.data import Dataset
from . import finetune_prompt, prompt_util
import random


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
LABEL_PAD_TOKEN_ID = -100


class ContentQualityDataset(Dataset):

    def __init__(self, tokenizer, config, train=False, init=True):
        self.data = []
        if init:
            if train:
                file = config.train_file
            else:
                file = config.eval_file
            with open(file, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        one_data = self.data[index]
        inputs, labels = self.process_using_api(one_data, self.tokenizer)
        # inputs, labels = self.process(one_data, self.tokenizer)
        return {
            "input_ids": inputs,
            "labels": labels
        }

    def process(self, data, tokenizer):
        # GPT4 origin prompt and result
        prompt = data["prompt"]
        gpt_result = json.dumps(data["gpt_result"])

        prompt_input = f"{B_INST} {prompt.strip()} {E_INST}"
        current_input = f"{prompt_input} {gpt_result.strip()} "
        current_input_ids = [tokenizer.bos_token_id] + \
                            tokenizer.encode(current_input, add_special_tokens=False, truncation=False) + \
                            [tokenizer.eos_token_id]
        prompt_input_ids = [tokenizer.bos_token_id] + \
                           tokenizer.encode(prompt_input, add_special_tokens=False, truncation=False)

        inputs = current_input_ids
        labels = copy.deepcopy(current_input_ids)
        for i in range(len(prompt_input_ids)):
            labels[i] = LABEL_PAD_TOKEN_ID

        assert len(inputs) == len(labels)
        inputs = inputs[:tokenizer.model_max_length]
        labels = labels[:tokenizer.model_max_length]
        return inputs, labels

    def process_using_api(self, data, tokenizer):
        prompt = data["prompt"]
        gpt_result = json.dumps(data["gpt_result"])
        return self.process_dialog([prompt, gpt_result], tokenizer)

    def process_without_rationale(self, data, tokenizer):
        title = data["title"]
        title = prompt_util.get_by_limit_len_title(title)
        content = data["content"]
        content = prompt_util.get_by_limit_len_content(content)
        prompt = finetune_prompt.writing_quality_without_rationale_prompt.format(title=title, content=content)
        gpt_result = prompt_util.remove_writing_quality_rational(data["gpt_result"])
        gpt_result = json.dumps(gpt_result)
        return self.process_dialog([prompt, gpt_result], tokenizer)

    def get_without_rational_prompt(self, data):
        title = "" if "title" not in data else data["title"]
        content = "" if "content" not in data else data["content"]
        prompt = finetune_prompt.writing_quality_without_rationale_prompt.format(title=title, content=content)
        return prompt

    def get_without_rational_gpt_result(self, data):
        gpt_result = {} if "gpt_result" not in data else data["gpt_result"]
        rational_list = ["Clarity_and_Coherence_Evaluation_Rationale", "Relevance_and_Focus_Evaluation_Rationale",
                         "Accuracy_and_Credibility_Evaluation_Rationale", "Originality_and_Insightfulness_Evaluation_Rationale",
                         "Engagement_and_Interest_Evaluation_Rationale", "Grammar_and_Style_Evaluation_Rationale",
                         "Structural_Integrity_Evaluation_Rationale", "Argumentation_and_Evidence_Evaluation_Rationale",
                         "Overall_Evaluation_Rationale"]
        for one_rational in rational_list:
            if one_rational in gpt_result:
                del gpt_result[one_rational]
        return json.dumps(gpt_result)

    def process_dialog(self, dialog, tokenizer, min_turn_idx=0, return_prompt=False, train=False, train_on_context=-1):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        assert len(dialog) >= 2
        dialog = dialog[:2 * len(dialog) // 2]
        inputs = []
        labels = []
        total_turns = len(dialog) // 2
        prompt = ""
        for turn_idx in range(total_turns):
            cur_turn_text = f"{B_INST} {dialog[turn_idx * 2].strip()} {E_INST} {dialog[turn_idx * 2 + 1].strip()} "

            turn_input = [tokenizer.bos_token_id] + \
                         tokenizer.encode(cur_turn_text,
                                          add_special_tokens=False,
                                          truncation=False) + \
                         [tokenizer.eos_token_id]
            if turn_idx >= min_turn_idx:
                cur_turn_only_input_text = f"{B_INST} {dialog[turn_idx * 2].strip()} {E_INST}"
                turn_only_input = tokenizer.encode(cur_turn_only_input_text,
                                                   add_special_tokens=False,
                                                   truncation=False)
                turn_label = turn_input.copy()
                input_len = len(turn_only_input) + 1
                for i in range(input_len):  # plus one for bos
                    if train and random.random() < train_on_context and i > input_len * 0.15 and i < input_len * 0.85:
                        # this will include some token in the prompt for loss
                        continue
                    turn_label[i] = IGNORE_INDEX
                prompt += cur_turn_only_input_text
            else:
                # for single turn training, we need to mask all history
                turn_label = [IGNORE_INDEX] * len(turn_input)
                prompt += cur_turn_text
            inputs.extend(turn_input)
            labels.extend(turn_label)
        if return_prompt:
            return prompt
        assert len(inputs) == len(labels)
        inputs = inputs[:tokenizer.model_max_length]
        labels = labels[:tokenizer.model_max_length]
        return inputs, labels
