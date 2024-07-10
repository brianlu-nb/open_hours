import json
import numpy as np
from datetime import datetime, timedelta
from datasets import load_dataset
from hours_prompts_db import get_prompt
from hours_prompts_db import PromptType as ptype
import generate_hours
import os

class Prompt:
    def __init__(self, opening_hours: dict, type: ptype, use_delta: bool):
        self.hours = opening_hours
        self.type = type
        self.use_delta = use_delta
        
        self.name = ""
        if not self.type == ptype.TO_LIST:
            self.name = np.random.choice(list(self.hours.keys()))
        
        precision = np.random.choice(['minute', 'hour'])
        if self.type == ptype.TO_HOURS:
            precision='day'
        self.today = random_time(precision=precision)
        
        self.delta = timedelta()
        if self.use_delta:
            self.delta = timedelta(days=np.random.randint(1, 8))
        self.q_time = self.today + self.delta
        
        self.user_prompt = get_prompt(self.type, self.name, self.q_time, self.use_delta, self.delta)
        self.expected_response = self._generate_expected_response()
      
    @staticmethod  
    def wrap_dict(input: dict):
        output = Prompt(
            opening_hours=input['opening_hours'],
            type=ptype[input['problem_type']],
            use_delta=input['use_delta']
        )
        output.today=datetime.strptime(input['today'], "%B %d, %Y")
        output.q_time=output.today
        output.user_prompt=input['user_prompt']
        output.expected_response=input['expected_response']
        return output
       
    def _generate_expected_response(self):
        def format_hours(val: str):
            return "Closed" if val == "" else val
        
        day = f"{self.q_time:%a}"
        out = {}
        if not self.name == "":
            out['name'] = self.name
            hours = self.hours[self.name][day]
            open = in_hours(self.q_time, hours)
            hours = format_hours(hours)
        else:
            hours = {}
            # open = []
            open = {}
            for key, val in self.hours.items():
                # if in_hours(self.q_time, val[day]):
                #     open.append(key)
                open[key] = in_hours(self.q_time, val[day])
                hours[key] = format_hours(val[day])
        
        out['time'] = f"{self.q_time:%Y-%m-%d %I:%M%p}"
        out['day_of_week'] = day
        out['opening_hours'] = hours
        
        if not self.type == ptype.TO_HOURS:
            out["is_open"] = open
        
        return out
    
    def __repr__(self):
        return json.dumps(self.to_dict_presentable(), indent=4)
    
    def to_dict(self):
        out = {
            "problem_type": self.type,
            "use_delta": self.use_delta,
            "today": self.today,
            "opening_hours": self.hours,
            "user_prompt": self.user_prompt,
            "expected_response": self.expected_response,
        }
        return out
    
    def to_dict_presentable(self):
        out = self.to_dict()
        out["today"] = f"{self.today:%B %d, %Y}"
        return out
        
    def evaluate_response(self, response: dict) -> float:
        if self.type == ptype.TO_LIST and 'is_open' in response:
            out = 0.0
            for name in self.hours:
                if name in response['is_open'] and response['is_open'][name] == self.expected_response['is_open'][name]:
                    out += 1.0 
            return out / len(self.hours)
        
        for key, val in self.expected_response.items():
            if key not in response or response[key] != val:
                return 0
        return 1

    def num_true_positives(self, response: dict):
        if self.type == ptype.TO_HOURS or self.type == ptype.TO_BOOL:
            return self.num_reported_positives(response) and self.num_expected_positives()
        if self.type == ptype.TO_LIST and 'is_open' in response:
            out = 0.0
            for name in self.hours:
                # if name in self.expected_response['is_open'] and name in response['is_open']:
                if self.expected_response['is_open'][name] and name in response['is_open'] and response['is_open'][name]:
                    out += 1.0 
            return out
        return 0
    
    def num_reported_positives(self, response: dict):
        d1 = self.expected_response
        d2 = response
        
        match self.type:
            case ptype.TO_BOOL:
                return int('is_open' in response and response['is_open'])
            case ptype.TO_HOURS:
                return int(elem_equal(d1, d2, "opening_hours"))
            case ptype.TO_LIST:
                return 0.0 if 'is_open' not in response else sum([val for _, val in response['is_open'].items()]) # len(response['is_open'])
            case _:
                return 0
    
    def num_expected_positives(self):
        match self.type:
            case ptype.TO_BOOL:
                return int(self.expected_response['is_open'])
            case ptype.TO_HOURS:
                return 1
            case ptype.TO_LIST:
                return sum([val for _, val in self.expected_response['is_open'].items()]) # len(self.expected_response['is_open'])
            case _:
                return 0
            
    def output_report(self, response: dict):
        tp, fp, fn, tn, nfp, nfn = 0, 0, 0, 0, 0, 0
        isopen = response['is_open']
        if self.type == ptype.TO_LIST:
            for key, exp_val in self.expected_response['is_open'].items():
                
                if exp_val == True:
                    if key not in isopen:
                        nfp += 1
                    elif isopen[key] == True:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if key not in isopen:
                        nfn += 1
                    elif isopen[key] == True:
                        fp += 1
                    else:
                        tn += 1
            return (tp, fp, fn, tn, nfp, nfn)
        
        total = 1
        tp = self.num_true_positives(response)
        fp = self.num_reported_positives(response) - tp
        fn = self.num_expected_positives() - tp
        tn = total - tp - fp - fn
        
        return (tp, fp, fn, tn, nfp, nfn)

    def new_keys(self, response: dict):
        if self.type != ptype.TO_LIST or 'is_open' not in response:
            return 0, 0
            
        out = [0, 0]
        for name, val in response['is_open'].items():
            if name not in self.expected_response['is_open']:
                out[0 if val else 1] += 1
        return out

# Dictionary Util

def elem_equal(d1: dict, d2: dict, key: str) -> bool:
    return (key in d1 and key in d2 and d1[key] == d2[key]) or (key not in d1 and key not in d2)

def contains_xor(l1: list, l2: list, elem) -> bool:
    return (elem in l1 and not elem in l2) or (elem in l2 and not elem in l1)

# Time Util

def random_time(min_time=datetime.now(), max_time=datetime(2030, 12, 31, 23, 59, 59, 999999), precision='minute') -> datetime:
    time = datetime.fromtimestamp(np.random.uniform(min_time.timestamp(), max_time.timestamp()))
    time = truncate_time(time, precision)
    return time

def truncate_time(time: datetime, precision='minute'):
    if precision == 'year':
        return datetime(time.year, 1, 1, 0, 0, 0)
    elif precision == 'month':
        return datetime(time.year, time.month, 1, 0, 0, 0)
    elif precision == 'day':
        return datetime(time.year, time.month, time.day, 0, 0, 0)
    elif precision == 'hour':
        return datetime(time.year, time.month, time.day, time.hour, 0, 0)
    elif precision == 'minute':
        return datetime(time.year, time.month, time.day, time.hour, time.minute, 0)
    elif precision == 'second':
        return datetime(time.year, time.month, time.day, time.hour, time.minute, time.second)
    else:
        raise ValueError(f"Invalid unit: {precision}")

def in_hours(time: datetime, hours: str) -> bool:
    if hours == '':
        return False

    for dur in hours.split(", "):
        start_time_str, end_time_str = dur.split("-")

        start_time = datetime.combine(time, datetime.strptime(start_time_str, "%I:%M%p").time())
        end_time = datetime.combine(time, datetime.strptime(end_time_str, "%I:%M%p").time())
        if start_time <= time < end_time:
            return True
    return False

# Generation!
def get_data(hours_dict: dict) -> dict:
    data = np.random.choice(list(hours_dict.keys()), np.random.randint(3,6))
    return {name: hours_dict[name] for name in data}

def generate_prompts(hours_data: dict, prompt_type: str, use_delta: bool, num_trials: int) -> list[Prompt]:
    prompt_list = []
    for _ in range(num_trials):
        data = get_data(hours_data)
        prompt_list.append(Prompt(data, prompt_type, use_delta))
    return prompt_list 

# Main

if __name__ == "__main__":
    with open(f"{generate_hours.current_path()}/hours.json", 'r') as file:
        hours_dict = json.load(file)
    prompts = generate_prompts(hours_dict, ptype.TO_LIST, False, 1)
    # new_data = []
    # for prompt in prompts:
    #     new_data.append({
    #         "opening_hours": prompt._data,
    #         "user_prompt": prompt.user_prompt,
    #         "output": prompt.expected_response,
    #     })
    
    # with open("data.jsonl", 'w') as file:
    #     file.writelines(f"{json.dumps(row)}\n" for row in new_data)
    for prompt in prompts:
        print(json.dumps(prompt.to_dict_presentable(), indent=4))