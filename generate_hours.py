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

def current_path():
    contents = os.listdir('test_hours/data')
    for item in contents:
        if item[0] == 'C':
            return f'test_hours/data/{item}'

    return ''

load_dotenv()
client = OpenAI()

def get_name_list(amt: int):
    remaining = amt
    output = set()
    with tqdm(range(amt), desc=f"Generating {amt} Names") as pbar:
        while remaining > 0:
            to_gen = 100 if remaining > 50 else remaining * 2
            if remaining > 1000:
                to_gen = remaining // 10
            if to_gen > 200:
                to_gen = 200
            pbar.desc = f"Generating {to_gen}/{amt} Names"  
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": "You are a creative name generator, outputting lists in json format."},
                    {"role": "user", "content": f"Generate {to_gen} restaurant names using only characters of the alphabet in the following format: {{'1': 'Name 1', '2': 'Name 2', ...}}"}
                ],
                temperature=1
            )
            num_update = 0
            
            def clean(s: str):
                s = s.strip()
                s = "'".join(s.split("\u2019"))
                s = "e".join(s.split("\u00e9"))
                return s
            
            try:
                output_list = [clean(n) for n in json.loads(response.choices[0].message.content).values()]
                output = set(output_list) | output
                
                num_update = len(output) - amt + remaining
                pbar.postfix = (f"Update: {num_update}")
                
                remaining = amt - len(output)
            except:
                pass
            pbar.update(num_update)
            
    
    output = list(output)[:amt]
    return output

def random_time(lower = time(0, 0), upper = time(hour=23, minute=59)):
    lower_minutes = lower.hour * 60 + lower.minute
    upper_minutes = upper.hour * 60 + upper.minute

    random_minutes = random.randint(lower_minutes, upper_minutes)
    hours, minutes = divmod(random_minutes, 60)

    precision = np.random.choice([60, 30, 15], p=[0.1, 0.7, 0.2])
    minutes = minutes // precision * precision
    return time(hours, minutes)

def random_interval(lower = time(0, 0), upper = time(23, 59)):
    t1, t2 = time(0,0), time(0,0)
    while t1 == t2:
        t1 = random_time(lower, upper)
        t2 = random_time(lower, upper)
        if t1 > t2:
            t1, t2 = t2, t1
    return f"{t1:%I:%M%p}-{t2:%I:%M%p}"

def random_daily_hours():
    rand = random.random()
    if rand < 0.1:
        return ""
    
    if rand < 0.6:
        return random_interval()
        
    split = time(random.randint(11, 16), 0)
    return f"{random_interval(upper=split)}, {random_interval(lower=split)}"

def random_weekly_hours():
    out = {}
    days = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
    for day in days:
        out[day] = random_daily_hours()
    return out

def generate_hours(num_entries: int):
    names = get_name_list(num_entries)
    hours_dict = {}
    with tqdm(range(len(names))) as pbar:
        for name in names:
            hours_dict[name] = random_weekly_hours()
            with open(f"{current_path()}/hours.json", 'w') as file:
                file.write(json.dumps(hours_dict, indent=4))
            pbar.update()
    return hours_dict

def generate_data(problem_type: ptype, use_delta: bool, num_entries: int, splits: tuple[float, float, float]):
    assert sum(splits) == 1

    cur_path = current_path()
    if cur_path != '':
        shutil.move(current_path(), f'test_hours/data/{current_path().split("Current_")[1]}')

    cur_path = f"test_hours/data/Current_Run_{datetime.now():%m-%d-%YT%H-%M-%S}"
    os.mkdir(cur_path)
    
    hours_dict = generate_hours(num_entries // 10)
    splits = [int(len(hours_dict) * sum(splits[:i])) for i in range(len(splits) + 1)]
    split_name = ["train", "test", "eval"]
    
    outputs = ["", "", ""]
    
    for i, left, right, split in zip(range(3), splits, splits[1:], split_name):
        hours_split = dict(list(hours_dict.items())[left:right])
        output = hours_prompts.generate_prompts(hours_split, problem_type, use_delta, (right - left) * 10)
        outputs[i] = output.copy()
        with open(f"{cur_path}/prompts_{split}.json", 'w') as file:
            printed = []
            for row in output:
                printed.append(row.to_dict_presentable())
            file.write(json.dumps(printed, indent=4))
    
    return outputs

def clean_data():
    cur_path = current_path()
    files = ["train.json", "test.json", "eval.json"]
    
    for file_name in files:
        with open(f"{cur_path}/prompts_{file_name}", 'r') as file:
            raw_data = json.load(file)
            jsonl_file = open(f"{cur_path}/{file_name}l", 'w')
            # new_data = []
            for line in raw_data:
                # if line['prompt']['expected_response']['day_of_week'] == line['full_response']['day_of_week']:
                #     line['prompt']['expected_response'] = line['full_response']
                #     new_data.append(line['prompt'])
                    # jsonl_file.write(f"{json.dumps(line['prompt'])}\n")
                jsonl_file.write(f"{json.dumps(line)}\n")

if __name__ == "__main__":
    # generate_data(ptype.TO_LIST, False, 64000, (0.8, 0.1, 0.1))
    clean_data()