from typing import Optional
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import hours_prompts
from hours_prompts import Prompt, generate_prompts, PromptType as ptype
import generate_hours
from huggingface_hub import InferenceClient

USE_GPT = True
DEBUG = False

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()
MODEL = 'gpt-4o'
TEMPERATURE = 1.0

def query(prompt: dict) -> str:
    if USE_GPT:
        with open('test_hours/hours_system_prompt.txt', 'r', encoding='utf-8') as file:
            system_prompt = file.read()
        with open('test_hours/hours_user_prompt.txt', 'r', encoding='utf-8') as file:
            user_prompt = file.read()
            user_prompt = user_prompt.format(
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
            temperature=TEMPERATURE,
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

def print_report(title: str, tally: dict, prompt: Prompt, response: Optional[dict] = None):
    tally['count'] += 1
    
    primary_keys = [
        'expected_positives',
        'expected_negatives',
    ]

    secondary_keys = [
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives',
        'reported_positives',
        'reported_negatives',
    ]

    if response:
        keys = primary_keys + secondary_keys
        for key in keys:
            tally[key] += getattr(prompt, key)(response)
        if prompt.correct(response):
            tally['correct'] += 1
    else:
        tally['prompt_errors'] += 1
        tally['individual_errors'] += len(prompt.hours)
        for key in primary_keys:
            tally[key] += getattr(prompt, key)({})

    total = tally['expected_positives'] + tally['expected_negatives']
    accuracy = (tally['true_positives'] + tally['true_negatives']) / total if total > 0 else 0

    def precision(tp, reported):
        return tp / reported if reported > 0 else 0

    def recall(tp, expected):
        return tp / expected if expected > 0 else 0

    def f1(precision, recall):
        return 2 / (1 / precision + 1 / recall) if precision * recall > 0 else 0

    positive_precision = precision(tally['true_positives'], tally['reported_positives'])
    positive_recall = recall(tally['true_positives'], tally['expected_positives'])
    negative_precision = precision(tally['true_negatives'], tally['reported_negatives'])
    negative_recall = recall(tally['true_negatives'], tally['expected_negatives'])

    positive_F1 = f1(positive_precision, positive_recall)
    negative_F1 = f1(negative_precision, negative_recall)

    positives_not_found = tally['expected_positives'] - tally['true_positives'] - tally['false_negatives']
    negatives_not_found = tally['expected_negatives'] - tally['false_positives'] - tally['true_negatives']
    total_not_found = positives_not_found + negatives_not_found

    positives_new_key = tally['reported_positives'] - tally['true_positives'] - tally['false_positives']
    negatives_new_key = tally['reported_negatives'] - tally['false_negatives'] - tally['true_negatives']
    total_new_keys = positives_new_key + negatives_new_key
    
    tally['postfix'] = {
        "Correct": tally['correct'],
        "Error": tally['prompt_errors'],
        "Accuracy": accuracy,
        "Precision": positive_precision,
        "Recall": positive_recall
    }

    if DEBUG:
        with open('test_hours/report_template.txt', 'r', encoding='utf-8') as file:
            out = file.read().format(
                title = title,
                tp = tally['true_positives'],
                tn = tally['true_negatives'],
                fp = tally['false_positives'],
                fn = tally['false_negatives'],
                total = total,
                ep = tally['expected_positives'],
                en = tally['expected_negatives'],
                rp = tally['reported_positives'],
                rn = tally['reported_negatives'],
                nfp = positives_not_found,
                nfn = negatives_not_found,
                rnf = total_not_found,
                nkp = positives_new_key,
                nkn = negatives_new_key,
                tnk = total_new_keys,
                pprec = positive_precision * 100,
                nprec = negative_precision * 100,
                precl = positive_recall * 100,
                nrecl = negative_recall * 100,
                pf1 = positive_F1 * 100,
                nf1 = negative_F1 * 100,
                acc = accuracy * 100,
                num_trials = tally['count'],
                correct = tally['correct'],
                err = tally['individual_errors'],
                correctp = tally['correct'] / tally['count'] * 100,
                errp = tally['individual_errors'] / tally['count'] * 100,
            )

        with open('test_hours/report.out', 'w', encoding='utf-8') as file:
            file.write(out)
        
        with open('test_hours/responses.jsonl', 'a', encoding='utf-8') as file:
            file.write(f'{json.dumps({"id": tally["count"], "prompt": prompt.to_dict_presentable(), "full_response": response}, ensure_ascii=False)}\n')

def run_prompts(prompts: list[Prompt]):
    keys = [
        'count',
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives',
        'expected_positives',
        'expected_negatives',
        'reported_positives',
        'reported_negatives',
        'prompt_errors',
        'individual_errors',
        'correct',
        'postfix'
    ]
    tally = {key: 0 for key in keys}
    
    with tqdm(desc=f'Querying', total=len(prompts)) as pbar:
        for prompt in prompts:
            response = query(prompt.to_dict())
            try:
                response_dict = json.loads(response)
            except json.JSONDecodeError as _:
                response_dict = None
            print_report('New Form of Output', tally, prompt, response_dict)
            pbar.set_postfix(tally['postfix'])
            pbar.update()
        


# runs


with open(f'{generate_hours.current_path()}/hours.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    prompts = generate_prompts(data, ptype.TO_LIST, False, 3, 5, 100)
    run_prompts(prompts)
# client = InferenceClient(model='http://0.0.0.0:8080')

# with open(f'test_hours/_correct.json', 'w', encoding='utf-8') as file:
#     file.write('[]')
# with open(f'test_hours/_incorrect.json', 'w', encoding='utf-8') as file:
#     file.write('[]')
# with open(f'test_hours/_hours.out', 'w', encoding='utf-8') as file:
#     file.write('')

# with open(f'{generate_hours.current_path()}/hours.json', 'r', encoding='utf-8') as file:
#     hours_dict = json.load(file)
# run_one_prompt(ptype.TO_LIST, False, 1000, 0)

# prompts = generate_prompts(prompt_type=ptype.TO_LIST, use_delta=False, num_trials=10000)
# with open('test_hours/data-2.jsonl', 'w', encoding='utf-8') as file:
#     file.writelines([f'{json.dumps(prompt.to_dict_presentable(), ensure_ascii=False)}\n' for prompt in prompts])