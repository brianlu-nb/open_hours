from typing import Optional
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
from hours_prompts import Prompt, generate_prompts, PromptType as ptype
import generate_hours
from huggingface_hub import InferenceClient
import prompt_templates
from datetime import datetime

from typing import Optional

class Report:
    def __init__(self):
        self._count = 0
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0
        self._ep = 0
        self._en = 0
        self._rp = 0
        self._rn = 0
        self._prompt_errors = 0
        self._individual_errors = 0
        self._correct = 0
    
    @property
    def true_positives(self):
        return self._tp
    @property
    def true_negatives(self):
        return self._tn
    @property
    def false_positives(self):
        return self._fp
    @property
    def false_negatives(self):
        return self._fn
    @property
    def expected_positives(self):
        return self._ep
    @property
    def expected_negatives(self):
        return self._en
    @property
    def reported_positives(self):
        return self._rp
    @property
    def reported_negatives(self):
        return self._rn
    @property
    def count(self):
        return self._count
    @property
    def correct(self):
        return self._correct
    @property
    def prompt_errors(self):
        return self._prompt_errors
    @property
    def individual_errors(self):
        return self._individual_errors
    @property
    def total(self):
        return self._ep + self._en
    @property
    def accuracy(self):
        return (self._tp + self._tn) / self.total if self.total > 0 else 0
    @property
    def positive_precision(self):
        return self._tp / self._rp if self._rp > 0 else 0
    @property
    def positive_recall(self):
        return self._tp / self._ep if self._ep > 0 else 0
    @property
    def positive_F1(self):
        precision = self.positive_precision
        recall = self.positive_recall
        return 2 * (precision * recall) / (precision + recall) if precision > 0 and recall > 0 else 0
    @property
    def negative_precision(self):
        return self._tn / self._rn if self._rn > 0 else 0
    @property
    def negative_recall(self):
        return self._tn / self._en if self._en > 0 else 0
    @property
    def negative_F1(self):
        precision = self.negative_precision
        recall = self.negative_recall
        return 2 * (precision * recall) / (precision + recall) if precision > 0 and recall > 0 else 0
    @property
    def positives_not_found(self):
        return self._ep - self._tp - self._fn
    @property
    def negatives_not_found(self):
        return self._en - self._tn - self._fp
    @property
    def total_not_found(self):
        return self.positives_not_found + self.negatives_not_found
    @property
    def positive_new_keys(self):
        return self._rp - self._tp - self._fp
    @property
    def negative_new_keys(self):
        return self._rn - self._tn - self._fn
    @property
    def total_new_keys(self):
        return self.positive_new_keys + self.negative_new_keys

    def update(self, prompt, response=None):
        self._count += 1
        self._ep += prompt.expected_positives()
        self._en += prompt.expected_negatives()
        if response:
            self._tp += prompt.true_positives(response)
            self._tn += prompt.true_negatives(response)
            self._fp += prompt.false_positives(response)
            self._fn += prompt.false_negatives(response)
            self._rp += prompt.reported_positives(response)
            self._rn += prompt.reported_negatives(response)
            if prompt.correct(response):
                self._correct += 1
        else:
            self._prompt_errors += 1
            self._individual_errors += len(prompt.hours)

    def report_str(self, title: Optional[str] = 'output'):
        with open('test_hours/report_template.txt', 'r', encoding='utf-8') as file:
            return file.read().format(
                title=title,
                tp=self.true_positives, tn=self.true_negatives, fp=self.false_positives, fn=self.false_negatives,
                ep=self.expected_positives, en=self.expected_negatives, rp=self.reported_positives, rn=self.reported_negatives,
                total=self.total,
                nfp=self.positives_not_found, nfn=self.negatives_not_found, rnf=self.total_not_found,
                nkp=self.positive_new_keys, nkn=self.negative_new_keys, tnk=self.total_new_keys,
                pprec=self.positive_precision * 100, precl=self.positive_recall * 100, pf1=self.positive_F1 * 100,
                nprec=self.negative_precision * 100, nrecl=self.negative_recall * 100, nf1=self.negative_F1 * 100,
                acc=self.accuracy * 100,
                num_trials=self.count,
                err=self.prompt_errors, errp=self.prompt_errors / self.count * 100 if self.count > 0 else 0,
                correct=self.correct, correctp=self.correct / self.count * 100 if self.count > 0 else 0,
            )

    def postfix(self):
        return {
            "Correct": self.correct,
            "Error": self.prompt_errors,
            "Accuracy": self.accuracy,
            "Precision": self.positive_precision,
            "Recall": self.positive_recall
        }

USE_GPT = False
DEBUG = True

if USE_GPT:
    load_dotenv()
    API_KEY = os.getenv('OPENAI_API_KEY')
    client = OpenAI()
    MODEL = 'gpt-4o'
else:
    client = InferenceClient(model="http://0.0.0.0:8080")

def query(prompt: dict) -> str:
    if USE_GPT:
        with open('test_hours/hours_system_prompt.txt', 'r', encoding='utf-8') as file:
            system_prompt = file.read()
            
        if prompt['use_delta']:
            user_prompt_template = prompt_templates.templates_timed[prompt['problem_type']]
        else:
            user_prompt_template = prompt_templates.templates[prompt['problem_type']]
        user_prompt = user_prompt_template.format(
                opening_hours=prompt['opening_hours'],
                today=f"{prompt['today']:%B %d, %Y}",
                user_prompt=prompt['user_prompt'],
        )
            
        response = client.chat.completions.create(
            model=MODEL,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content
    
    with open('test_hours/prompt_without_reason.txt', 'r', encoding='utf-8') as file:
        prompt = file.read().format(
            opening_hours=prompt['opening_hours'],
            today=prompt['today'],
            user_prompt=prompt['user_prompt'],
        )
    output_text = client.text_generation(
        prompt,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.01,
        repetition_penalty=1.3,
        top_p=0.95,
        top_k=40,
        details=True,
    ).generated_text
    
    return output_text[output_text.find('{'):output_text.rfind('}') + 1]

def run_prompts(prompts: list[Prompt]):
    tally = Report()
    
    if DEBUG:
        debug_reports = {
            ptype.TO_BOOL: Report(),
            ptype.TO_HOURS: Report(),
            ptype.TO_LIST: Report()
        }
        debug_reports_timed = {
            ptype.TO_BOOL: Report(),
            ptype.TO_HOURS: Report(),
            ptype.TO_LIST: Report()
        }
    
    with tqdm(desc=f'Querying', total=len(prompts)) as pbar:
        for prompt in prompts:
            response = query(prompt.to_dict())
            try:
                response = json.loads(response)
            except json.JSONDecodeError as _:
                response = None
            
            tally.update(prompt, response)
            if DEBUG:
                if prompt.use_delta:
                    debug_reports_timed[prompt.type].update(prompt, response)
                else:
                    debug_reports[prompt.type].update(prompt, response)
                with open('test_hours/report.out', 'w', encoding='utf-8') as file:
                    file.write(f"{tally.report_str(f'Output as of {datetime.now():%B %d, %Y at %H:%M:%S}')}\n\n")
                    for type in list(ptype):
                        file.write(f"{debug_reports[type].report_str(f'----- {type}, False -------')}\n\n")
                        file.write(f"{debug_reports_timed[type].report_str(f'----- {type}, True ------')}\n\n")
                with open('test_hours/responses.jsonl', 'a', encoding='utf-8') as file:
                    file.write(f'{json.dumps({"id": tally.count, "prompt": prompt.to_dict_presentable(), "full_response": response}, ensure_ascii=False)}\n')

            pbar.set_postfix(tally.postfix())
            pbar.update()

# runs
if __name__ == "__main__":
    with open(f'{generate_hours.current_path()}/hours.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        prompt_lists = [generate_prompts(data, min_entries=3, max_entries=5, num_trials=1000)]
        # print(prompts[0].expected_response)
        for prompts in prompt_lists:
            run_prompts(prompts)