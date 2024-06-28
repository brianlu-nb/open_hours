import numpy as np
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import hours_prompts
from hours_prompts import Prompt, generate_prompts
from hours_prompts_db import PromptType as ptype
from huggingface_hub import InferenceClient



def query(qa: dict) -> str:
    # with open("test_hours/hours_system_prompt.txt", 'r') as file:
    #     system_prompt = file.read()
    # with open("test_hours/hours_user_prompt.txt", 'r') as file:
    #     user_prompt = file.read().format(
    #         opening_hours=qa['opening_hours'],
    #         today=f"{qa['today']:%B %d, %Y}",
    #         user_prompt=qa['user_prompt'],
    #     )
        
    # response = client.chat.completions.create(
    #     model=MODEL,
    #     response_format={ "type": "json_object" },
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ],
    #     temperature=TEMPERATURE,
    # )
    # return response.choices[0].message.content
    
    # schema = {
    #     "type": "json",
    #     "value": {
    #         "properties": {
    #             "time": {
    #                 "type": "string"
    #             },
    #             "day_of_week": {
    #                 "type": "string"
    #             },
    #             "opening_hours": {
    #                 "type": "string"
    #             },
    #             "is_open": {
    #                 "type": "array",
    #                 "items": {
    #                     "type": "string"
    #                 }
    #             }
    #         },
    #         "required": ["time", "day_of_week", "opening_hours", "is_open"]
    #     }
    # }
    with open("test_hours/prompt.txt", 'r') as file:
        prompt = file.read().format(
            opening_hours=qa['opening_hours'],
            today=qa['today'],
            user_prompt=qa['user_prompt'],
        )
    output_text = client.text_generation(
        prompt,
        max_new_tokens=1000,
        do_sample=True,
        temperature=1.0,
        repetition_penalty=1,
        top_p=0.95,
        top_k=40,
        details=True,
        # grammar=schema
    ).generated_text
    
    return output_text

def is_correct(response_dict: dict, output: dict) -> bool:
    for key in output:
        if key not in response_dict or response_dict[key] != output[key]:
            return False
    return True

def evaluate(id: int, response: str, prompt: Prompt, correct_ds: list, incorrect_ds: list) -> tuple[float, float, float, float, float]:
    with open('test_hours/response-2.out', 'w') as file:
        file.write(response)
    
    res_str = response
    correct = 0
    num_tp = 0
    num_positives = 0
    num_expected_positives = prompt.num_expected_positives()
    num_errors = 0
    try:
        response_dict = json.loads(response)
        correct = prompt.evaluate_response(response_dict)
        # print(f"prompt: {user_prompt}")
        # print(f"response: {response}")
        # print(f"{is_correct}")
        
        num_tp = prompt.num_true_positives(response_dict)
        num_positives = prompt.num_reported_positives(response_dict)
        res_str = response_dict
    except:
        num_errors += 1
        
    
    precision = num_tp / num_positives if num_positives != 0 else ''
    recall = num_tp / num_expected_positives if num_expected_positives != 0 else ''
    prec_str = 'NaN' if precision == '' else f"{precision:.2f}"
    rec_str = 'NaN' if recall == '' else f"{recall:.2f}"
    
    datum = {
        "id": id,
        "prompt": prompt.to_dict_presentable(),
        "full_response": res_str,
    }
    correct_ds.append(datum) if correct == 1 else incorrect_ds.append(datum)
    
    with open('test_hours/_hours-2.out', 'a') as file:
        file.write(f'Run {id}: {{Accuracy: {correct:.2f}, Precision: {prec_str}, Recall: {rec_str}}}\n')
    return correct, num_tp, num_positives, num_expected_positives, num_errors

def report(num_trials: int, num_correct: int, correct_ds: list, incorrect_ds: list):
    with open(f"test_hours/_correct-2.json", 'w') as f_correct:
        json.dump(correct_ds, f_correct, indent=4)
    
    with open(f"test_hours/_incorrect-2.json", 'w') as f_incorrect:
        json.dump(incorrect_ds, f_incorrect, indent=4)
    
    with open(f'test_hours/_hours-2.out', 'a') as file:
        file.write(f'----------------{num_correct}/{num_trials}----------------\n\n')
        
def run_one_prompt(prompt_type: ptype, use_delta: bool, num_trials: int, num_completed: int):
    print(f'Type: {prompt_type}, Delta: {use_delta}')
    correct_ds = []
    with open(f"test_hours/_correct-2.json", 'r') as file:
        correct_ds = json.load(file)
    
    incorrect_ds = []
    with open(f"test_hours/_incorrect-2.json", 'r') as file:
        incorrect_ds = json.load(file)

    prompt_list = generate_prompts(prompt_type, use_delta, num_trials)
    
    num_correct = 0
    num_accurate = 0
    true_positives = 0
    reported_positives = 0
    expected_positives = 0
    num_errors = 0

    with tqdm(desc=f"Querying", total=num_trials) as t:
        for i in range(num_trials):
            prompt = prompt_list[i]
            response = query(prompt.to_dict())
            
            accuracy, num_tp, num_p, num_ep, num_err = evaluate(i + num_completed, response, prompt, correct_ds, incorrect_ds)
            num_accurate += accuracy
            if accuracy == 1:
                num_correct += 1

            true_positives += num_tp
            reported_positives += num_p
            expected_positives += num_ep
            num_errors += num_err

            if (i + 1) % 10 == 0:
                report(i + 1, num_correct, correct_ds, incorrect_ds)

            t.postfix = (
                f"# Correct: {num_correct}, "
                f"# Error: {num_errors}, "
                f"Accuracy: {(num_accurate / (i + 1) * 100):.2f}%, "
                f"Precision: {(true_positives / (reported_positives + 1e-10) * 100):.2f}%, "
                f"Recall: {(true_positives / (expected_positives + 1e-10) * 100):.2f}%"
            )
            t.update()

    print()
    report(num_trials, num_correct, correct_ds, incorrect_ds)

# runs

MODEL = "gpt-4o"
TEMPERATURE = 1.0

load_dotenv()
# api_key = os.getenv('OPENAI_API_KEY')
# client = OpenAI()
client = InferenceClient(model="http://0.0.0.0:8080")

with open(f"test_hours/_correct-2.json", 'w') as file:
    file.write('[]')
with open(f"test_hours/_incorrect-2.json", 'w') as file:
    file.write('[]')
with open(f"test_hours/_hours-2.out", 'w') as file:
    file.write('')


run_one_prompt(ptype.TO_LIST, False, 1000, 0)

# prompts = generate_prompts(prompt_type=ptype.TO_LIST, use_delta=False, num_trials=10000)
# with open("test_hours/data-2.jsonl", 'w') as file:
#     file.writelines([f"{json.dumps(prompt.to_dict_presentable())}\n" for prompt in prompts])