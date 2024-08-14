from datetime import datetime, timedelta
import json
from enum import Enum
from typing import Optional, Literal
import numpy as np
from multipledispatch import dispatch

PROMPTS_FILE_PATH = 'open_hours/hours_prompts.json'

class NoValidPromptsError(Exception):
    pass

class PromptType(str, Enum):
    TO_BOOL = 'TO_BOOL'
    TO_HOURS = 'TO_HOURS'
    TO_LIST = 'TO_LIST'

def random_time(min_time: Optional[datetime] = None, max_time: Optional[datetime] = None, precision: str = 'minute') -> datetime:
    min_time = min_time or datetime.now()
    max_time = max_time or datetime.now() + timedelta(days=3650)
    
    time = datetime.fromtimestamp(np.random.uniform(min_time.timestamp(), max_time.timestamp()))
    time = truncate_time(time, precision)
    return time

def truncate_time(t: datetime, precision: Literal['year', 'month', 'day', 'hour', 'minute', 'second'] = 'minute'):
    precision_map = {
        'year': datetime(t.year, 1, 1, 0, 0, 0),
        'month': datetime(t.year, t.month, 1, 0, 0, 0),
        'day': datetime(t.year, t.month, t.day, 0, 0, 0),
        'hour': datetime(t.year, t.month, t.day, t.hour, 0, 0),
        'minute': datetime(t.year, t.month, t.day, t.hour, t.minute, 0),
        'second': datetime(t.year, t.month, t.day, t.hour, t.minute, t.second),
    }
    
    if precision not in precision_map:
        raise ValueError(f'Invalid precision: {precision}')
    
    return precision_map[precision]

def in_hours(time: datetime, hours: str) -> bool:
    if not hours:
        return False

    for time_range in hours.split(', '):
        try:
            start_time_str, end_time_str = time_range.split('-')
            start_time = datetime.combine(time.date(), datetime.strptime(start_time_str, '%I:%M%p').time())
            end_time = datetime.combine(time.date(), datetime.strptime(end_time_str, '%I:%M%p').time())
            if start_time <= time < end_time:
                return True
        except ValueError:
            # Handle the case where the time_range format is incorrect
            raise ValueError(f'Invalid time range format: {time_range}')
    return False

def load_prompt_data() -> dict:
    with open(PROMPTS_FILE_PATH, 'r', encoding='utf-8') as file:
        prompts_db = json.load(file)
    if not prompts_db:
        raise NoValidPromptsError('The prompts file must contain valid data.')
    return prompts_db

def generate_random_time_format(time: datetime) -> str:
    formats = [
        f'{time:%H:%M:%S}', f'{time:%H:%M}',
        f'{time:%I:%M:%S%p}', f'{time:%I:%M%p}',
        f'{time:%-I:%M:%S%p}', f'{time:%-I:%M%p}'
    ]
    if time.minute == 0 and np.random.rand() < 0.5:
        return f'{time:%-I%p}'
    return np.random.choice(formats)

def generate_random_datetime_format(time: datetime) -> str:
    dates = [
        f'{time:%Y-%m-%d}', f'{time:%Y/%m/%d}',
        f'{time:%m-%d-%Y}', f'{time:%m/%d/%Y}',
    ]
    transitions = [' ']
    if np.random.rand() < 0.5:
        dates += [
            f'{time:%B %d, %Y}', f'{time:%b %d, %Y}',
            f'{time:%A %B %d, %Y}', f'{time:%A %b %d, %Y}',
            f'{time:%a %B %d, %Y}', f'{time:%a %b %d, %Y}',
            f'{time:%A}', f'{time:%a}',
        ]
        transitions += [' at ']
    return f'{np.random.choice(dates)}{np.random.choice(transitions)}{generate_random_time_format(time)}'

def generate_prompt(prompt_type: PromptType, restaurant_name: str, time: datetime, use_delta: bool = False, delta: timedelta = timedelta()) -> str:
    prompts_db = load_prompt_data()
    rand = np.random.rand()
    
    if use_delta:
        prompts_key = f'{prompt_type.name}_DELTA'
        if prompts_key not in prompts_db:
            raise NoValidPromptsError(f'No prompts available for type: {prompt_type.name} with delta.')
        
        prompts_db = prompts_db[prompts_key]
        prompts = prompts_db.get('-1', [])
        if delta.days == 1:
            prompts = prompts_db.get('1', [])
        elif str(delta.days) in prompts_db and rand < 0.5:
            prompts = prompts_db.get(str(delta.days), [])
            
        prompt = np.random.choice(prompts)
    else:
        prompts = prompts_db.get(prompt_type.name, [])
    
    if not prompts:
        raise NoValidPromptsError(f'No prompts available for type: {prompt_type.name} with delta: {use_delta}')
    
    prompt = np.random.choice(prompts)
    random_time_format = generate_random_time_format(time)
    random_datetime_format = generate_random_datetime_format(time)
  
    return prompt.format(
        restaurant_name=restaurant_name,
        time=time,
        random_time_format=random_time_format,
        random_datetime_format=random_datetime_format,
        days=delta.days
    )

# Prompt Class
class Prompt:
    def __init__(self, opening_hours: dict, type: PromptType, use_delta: bool):
        self.hours = opening_hours
        self.type = type
        self.use_delta = use_delta
        
        self.name = ''
        if self.type != PromptType.TO_LIST:
            self.name = np.random.choice(list(self.hours.keys()))
        
        precision = 'day' if self.type == PromptType.TO_HOURS else np.random.choice(['minute', 'hour'])
        self.today = random_time(precision=precision)
        
        self.delta = timedelta(days=np.random.randint(1, 8)) if self.use_delta else timedelta()
        self.q_time = self.today + self.delta
        
        self.user_prompt = generate_prompt(self.type, self.name, self.q_time, self.use_delta, self.delta)
        self.expected_response = self._generate_expected_response()
      
    @staticmethod  
    def from_dict(input: dict) -> 'Prompt':
        output = Prompt(
            opening_hours=input['opening_hours'],
            type=PromptType[input['problem_type']],
            use_delta=input['use_delta']
        )
        output.today=datetime.strptime(input['today'], '%B %d, %Y')
        output.q_time=output.today
        output.user_prompt=input['user_prompt']
        output.expected_response=input['expected_response']
        return output
       
    def _generate_expected_response(self) -> dict:
        def format_hours(val: str) -> str:
            return 'Closed' if val == '' else val
        
        day = f'{self.q_time:%a}'
        out = {}
        
        if self.name:
            out['name'] = self.name
            hours = self.hours[self.name][day]
            open = in_hours(self.q_time, hours)
            hours = format_hours(hours)
        else:
            hours = {key: format_hours(val[day]) for key, val in self.hours.items()}
            open = {key: in_hours(self.q_time, val[day]) for key, val in self.hours.items()}
        
        out.update({
            'time': f'{self.q_time:%Y-%m-%d %I:%M%p}',
            'day_of_week': day,
            'opening_hours': hours,
        })
        
        if self.type != PromptType.TO_HOURS:
            out['is_open'] = open
        
        return out
    
    def _calculate_matches(self, response: dict, expected_val: bool, actual_val: bool) -> int:
        if self.type == PromptType.TO_BOOL:
            return int(
                'is_open' in response and 
                self.expected_response['is_open'] == expected_val and
                response['is_open'] == actual_val
            )
        
        if self.type == PromptType.TO_HOURS:
            return int(
                'opening_hours' in response and
                (self.expected_response['opening_hours'] == response['opening_hours']) == actual_val 
            ) if expected_val else 0
        
        if self.type == PromptType.TO_LIST and 'is_open' in response:
            return sum(
                1 for name in self.hours if (
                    self.expected_response['is_open'][name] == expected_val and
                    name in response['is_open'] and
                    response['is_open'][name] == actual_val
            ))
        
        return 0

    def _calculate_values(self, response: dict, expected_val: Optional[bool] = None, actual_val: Optional[bool] = None, reported: bool = False) -> int:
        if self.type == PromptType.TO_BOOL:
            if reported:
                return int('is_open' in response and response['is_open'] == actual_val)
            return int(self.expected_response['is_open'] == expected_val)
        
        if self.type == PromptType.TO_HOURS:
            if reported:
                return int('opening_hours' in response and (response['opening_hours'] == self.expected_response['opening_hours']) == actual_val)
            return int(expected_val)
        
        if self.type == PromptType.TO_LIST:
            if reported:
                return sum(1 for val in response['is_open'].values() if val == actual_val) if 'is_open' in response else 0
            return sum(1 for val in self.expected_response['is_open'].values() if val == expected_val)
        
        return 0

    def true_positives(self, response: dict) -> int:
        return self._calculate_matches(response, expected_val=True, actual_val=True)
    
    def true_negatives(self, response: dict) -> int:
        return self._calculate_matches(response, expected_val=False, actual_val=False)
    
    def false_positives(self, response: dict) -> int:
        return self._calculate_matches(response, expected_val=False, actual_val=True)
    
    def false_negatives(self, response: dict) -> int:
        return self._calculate_matches(response, expected_val=True, actual_val=False)
    
    def reported_positives(self, response: dict) -> int:
        return self._calculate_values(response, actual_val=True, reported=True)

    def reported_negatives(self, response: dict) -> int:
        return self._calculate_values(response, actual_val=False, reported=True)

    def expected_positives(self, response: Optional[dict] = None) -> int:
        return self._calculate_values({}, expected_val=True)

    def expected_negatives(self, response: Optional[dict] = None) -> int:
        return self._calculate_values({}, expected_val=False)

    def correct(self, response: dict) -> bool:
        if self.type == PromptType.TO_HOURS:
            try:
                return self.expected_response['opening_hours'] == response['opening_hours']
            except:
                return False
        try:
            return self.expected_response['is_open'] == response['is_open']
        except:
            return False

    def to_dict(self) -> dict:
        out = {
            "problem_type": self.type,
            "use_delta": self.use_delta,
            "today": self.today,
            "opening_hours": self.hours,
            "user_prompt": self.user_prompt,
            "expected_response": self.expected_response,
        }
        return out
    
    def to_dict_presentable(self) -> dict:
        out = self.to_dict()
        out.update({"today": f"{self.today:%B %d, %Y}"})
        return out

    def __repr__(self) -> str:
        return json.dumps(self.to_dict_presentable(), indent=4, ensure_ascii=False)
    
def get_data(hours_dict: dict, min_entries: int, max_entries: int) -> dict:
    if min_entries < 1 or max_entries < min_entries:
        raise ValueError('Invalid min_entries or max_entries values')

    selected_keys = np.random.choice(list(hours_dict.keys()), np.random.randint(min_entries, max_entries + 1), replace=False)
    return {name: hours_dict[name] for name in selected_keys}

def generate_prompts(hours_data: dict, prompt_type: Optional[PromptType] = None, use_delta: Optional[bool] = None, min_entries: int = 1, max_entries: int = 1, num_trials: int = 1) -> list[Prompt]:
    prompt_list = []
    for ptype in list(PromptType) if prompt_type is None else [prompt_type]:
        for delta in [True, False] if use_delta is None else [use_delta]:
            prompt_list.append((ptype, delta))
    output = []
    for _ in range(num_trials):
        data = get_data(hours_data, min_entries, max_entries)
        prompt_type, use_delta = prompt_list[np.random.randint(0, len(prompt_list))]
        output.append(Prompt(data, prompt_type, use_delta))
    return output

def generate_prompts_with_list(hours_data: dict, prompt_list: list, min_entries: int = 1, max_entries: int = 1, num_trials: int = 1) -> list[Prompt]:
    output = []
    for _ in range(num_trials):
        data = get_data(hours_data, min_entries, max_entries)
        prompt_type, use_delta = np.random.choice(prompt_list)
        output.append(Prompt(data, prompt_type, use_delta))
    return output
