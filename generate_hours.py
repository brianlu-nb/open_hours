import os
import shutil
import json
from datetime import datetime, time
import random
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import hours_prompts
from hours_prompts import generate_prompts, PromptType as ptype
from typing import Optional

# Constants
PROMPTS_FILE_PATH = 'open_hours/hours_prompts.json'
DATA_DIR = 'open_hours/data'

# Load environment variables and initialize OpenAI client
load_dotenv()
client = OpenAI()

def current_path() -> str:
    try:
        contents = os.listdir(DATA_DIR)
        for item in contents:
            if item.startswith('C'):
                return os.path.join(DATA_DIR, item)
    except FileNotFoundError:
        pass
    return ''

def get_name_list(amt: int) -> list:
    remaining = amt
    output = set()
    with tqdm(total=amt, desc=f'Generating {amt} Names') as pbar:
        while remaining > 0:
            if remaining > 2000:
                to_gen = 200
            elif remaining > 1000:
                to_gen = remaining // 10
            elif remaining > 100:
                to_gen = 100
            else:
                to_gen = remaining * 2
            response = client.chat.completions.create(
                model='gpt-4o',
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": "You are a creative name generator, outputting lists in json format."},
                    {"role": "user", "content": f"Generate {to_gen} restaurant names in the following format: {{'1': 'Name 1', '2': 'Name 2', ...}}"}
                ],
                temperature=1
            )
            
            try:
                output_list =[name.strip() for name in json.loads(response.choices[0].message.content).values()]
                initial_len = len(output)
                output.update(output_list)
                num_update = len(output) - initial_len
                pbar.set_postfix({"Update": num_update, "Prompted": to_gen})
                pbar.update(num_update)
                remaining = amt - len(output)
            except json.JSONDecodeError:
                pass
            pbar.update(0)
    
    return list(output)[:amt]

def random_time(lower: time = time(0, 0), upper: time = time(23, 59)) -> time:
    lower_minutes = lower.hour * 60 + lower.minute
    upper_minutes = upper.hour * 60 + upper.minute
    random_minutes = random.randint(lower_minutes, upper_minutes)
    hours, minutes = divmod(random_minutes, 60)
    precision = np.random.choice([60, 30, 15], p=[0.1, 0.7, 0.2])
    minutes = (minutes // precision) * precision
    return time(hours, minutes)

def random_interval(lower: time = time(0, 0), upper: time = time(23, 59)) -> str:
    t1, t2 = lower, lower
    while t1 == t2:
        t1, t2 = sorted([random_time(lower, upper), random_time(lower, upper)])
    return f'{t1:%I:%M%p}-{t2:%I:%M%p}'

def random_daily_hours() -> str:
    rand = random.random()
    if rand < 0.1:
        return ''
    if rand < 0.6:
        return random_interval()
    split = time(random.randint(11, 16), 0)
    return f'{random_interval(upper=split)}, {random_interval(lower=split)}'

def random_weekly_hours() -> dict:
    return {day: random_daily_hours() for day in ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')}

def generate_hours(num_entries: int) -> dict:
    names = get_name_list(num_entries)
    hours_dict = {}
    cur_path = current_path()
    os.makedirs(cur_path, exist_ok=True)
    
    with tqdm(total=len(names), desc='Generating Hours') as pbar:
        for name in names:
            hours_dict[name] = random_weekly_hours()
            with open(f'{cur_path}/hours.json', 'w', encoding='utf-8') as file:
                json.dump(hours_dict, file, indent=4, ensure_ascii=False)
            pbar.update()
    return hours_dict

def generate_data(num_entries: int, splits: tuple[float, float, float], problem_type: Optional[ptype] = None, use_delta: Optional[bool] = None) -> list:
    assert sum(splits) == 1

    cur_path = current_path()
    if cur_path:
        shutil.move(cur_path, f"open_hours/data/{cur_path.split('Current_')[1]}")

    cur_path = f'open_hours/data/Current_Run_{datetime.now():%Y_%m_%d_%H_%M_%S}'
    os.mkdir(cur_path)
    
    hours_dict = generate_hours(num_entries // 10)
    with open(f'{cur_path}/hours.json', 'r') as file:
        hours_dict = json.load(file)
    split_indices = [int(len(hours_dict) * sum(splits[:i])) for i in range(len(splits) + 1)]
    split_names = ['train', 'test', 'eval']
    outputs = ['', '', '']
    
    for i, left, right, split in zip(range(3), split_indices, split_indices[1:], split_names):
        hours_split = dict(list(hours_dict.items())[left:right])
        output = generate_prompts(hours_split, problem_type, use_delta, 5, 10, (right - left) * 10)
        outputs[i] = output.copy()
        with open(f'{cur_path}/prompts_{split}.json', 'w', encoding='utf-8') as file:
            json.dump([row.to_dict_presentable() for row in output], file, indent=4, ensure_ascii=False)


    files = ['train.json', 'test.json', 'eval.json']

    for file_name in files:
        with open(f'{cur_path}/prompts_{file_name}', 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
            with open(f'{cur_path}/{file_name}l', 'w', encoding='utf-8') as jsonl_file:
                for line in raw_data:
                    jsonl_file.write(f'{json.dumps(line, ensure_ascii=False)}\n')

    return outputs

if __name__ == "__main__":
    generate_data(100, (0.8, 0.1, 0.1))
