import os
import shutil
import json
from datetime import datetime, timedelta, time
import random
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import hours_prompts
from hours_prompts import PromptType as ptype
from typing import Optional

# Constants
PROMPTS_FILE_PATH = 'test_hours/hours_prompts.json'
DATA_DIR = 'test_hours/data'

# Load environment variables and initialize OpenAI client
load_dotenv()
client = OpenAI()

def current_path() -> str:
    """
    Get the current path in the 'test_hours/data' directory that starts with 'C'.
    
    Returns:
        str: The full path of the current directory starting with 'C'. Returns an empty string if not found.
    """
    try:
        contents = os.listdir(DATA_DIR)
        for item in contents:
            if item.startswith('C'):
                return os.path.join(DATA_DIR, item)
    except FileNotFoundError:
        return ''
    return ''

def get_name_list(amt: int) -> list:
    """
    Generate a list of unique restaurant names using OpenAI's API.
    
    Args:
        amt (int): The number of names to generate.
    
    Returns:
        list: A list of generated restaurant names.
    """
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
    """
    Generate a random time within the given bounds.

    Args:
        lower (time): The lower bound time. Defaults to 00:00.
        upper (time): The upper bound time. Defaults to 23:59.

    Returns:
        time: A randomly generated time.
    """
    lower_minutes = lower.hour * 60 + lower.minute
    upper_minutes = upper.hour * 60 + upper.minute
    random_minutes = random.randint(lower_minutes, upper_minutes)
    hours, minutes = divmod(random_minutes, 60)
    precision = np.random.choice([60, 30, 15], p=[0.1, 0.7, 0.2])
    minutes = (minutes // precision) * precision
    return time(hours, minutes)

def random_interval(lower: time = time(0, 0), upper: time = time(23, 59)) -> str:
    """
    Generate a random time interval within the given bounds.

    Args:
        lower (time): The lower bound time. Defaults to 00:00.
        upper (time): The upper bound time. Defaults to 23:59.

    Returns:
        str: A formatted time interval.
    """
    t1, t2 = lower, lower
    while t1 == t2:
        t1, t2 = sorted([random_time(lower, upper), random_time(lower, upper)])
    return f'{t1:%I:%M%p}-{t2:%I:%M%p}'

def random_daily_hours() -> str:
    """
    Generate random daily hours for a restaurant.

    Returns:
        str: A string representing the daily hours.
    """
    rand = random.random()
    if rand < 0.1:
        return ''
    if rand < 0.6:
        return random_interval()
    split = time(random.randint(11, 16), 0)
    return f'{random_interval(upper=split)}, {random_interval(lower=split)}'

def random_weekly_hours() -> dict:
    """
    Generate random weekly hours for a restaurant.

    Returns:
        dict: A dictionary representing the weekly hours.
    """
    return {day: random_daily_hours() for day in ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')}

def generate_hours(num_entries: int) -> dict:
    """
    Generate a dictionary of restaurant names and their weekly hours.

    Args:
        num_entries (int): The number of restaurant entries to generate.

    Returns:
        dict: A dictionary of restaurant names and their weekly hours.
    """
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
    """
    Generate data and split it into training, testing, and evaluation sets.

    Args:
        problem_type (PromptType): The type of problem for which to generate prompts.
        use_delta (bool): Whether to use a time delta for the prompts.
        num_entries (int): The number of entries to generate.
        splits (tuple): The split ratios for training, testing, and evaluation sets.

    Returns:
        list: A list of generated data splits.
    """
    assert sum(splits) == 1

    cur_path = current_path()
    if cur_path:
        shutil.move(cur_path, f"test_hours/data/{cur_path.split('Current_')[1]}")

    cur_path = f'test_hours/data/Current_Run_{datetime.now():%Y_%m_%d_%H_%M_%S}'
    os.mkdir(cur_path)
    
    hours_dict = generate_hours(num_entries // 10)
    split_indices = [int(len(hours_dict) * sum(splits[:i])) for i in range(len(splits) + 1)]
    split_names = ['train', 'test', 'eval']
    outputs = ['', '', '']
    
    for i, left, right, split in zip(range(3), split_indices, split_indices[1:], split_names):
        hours_split = dict(list(hours_dict.items())[left:right])
        output = hours_prompts.generate_prompts(
            hours_split, 
            prompt_type=problem_type, 
            use_delta=use_delta, 
            min_entries=5, 
            max_entries=10, 
            num_trials=(right - left) * 10
        )
        outputs[i] = output.copy()
        with open(f'{cur_path}/prompts_{split}.json', 'w', encoding='utf-8') as file:
            json.dump([row.to_dict_presentable() for row in output], file, indent=4, ensure_ascii=False)
    
    return outputs

def clean_data():
    """
    Clean and convert data from JSON to JSONL format.

    Returns:
        None
    """
    cur_path = current_path()
    files = ['train.json', 'test.json', 'eval.json']
    
    for file_name in files:
        with open(f'{cur_path}/prompts_{file_name}', 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
            with open(f'{cur_path}/{file_name}l', 'w', encoding='utf-8') as jsonl_file:
                for line in raw_data:
                    jsonl_file.write(f'{json.dumps(line, ensure_ascii=False)}\n')

if __name__ == "__main__":
    generate_data(64000, (0.8, 0.1, 0.1))
    clean_data()
