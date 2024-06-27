from datetime import datetime, timedelta
import numpy as np
from enum import Enum

class PromptType(str, Enum):
    TO_BOOL = 'toBool'
    TO_HOURS = 'toHours'
    TO_LIST = 'toList'

def get_prompt(problem_type: PromptType, res_name: str, final_time: datetime, use_timedelta=False, time_delta=timedelta()):
    t = random_time_format(final_time)
    r = np.random.rand()
    if problem_type == PromptType.TO_BOOL and not use_timedelta:
        return np.random.choice([
            f"Does {res_name} serve customers at {final_time:%Y-%m-%d} {t} GMT?",
            f"If I visit {res_name} at {final_time:%B %d, %Y %H:%M} {t}, will it be open?",
            f"If I go to {res_name} on {final_time:%b %d, %Y} at {t}, will it be open?",
            f"Is {res_name} open on {final_time:%A, %B %d, %Y} at {t}?",
            f"Can I visit {res_name} on {final_time:%Y-%m-%d} at {t}?",
            f"Checking if {res_name} is open on {final_time:%d %B %Y} at {t}.",
            f"Will {res_name} be operational on {final_time:%b %d, %Y} around {t}?",
            f"I'd like to know if {res_name} will be open at {t} on {final_time:%b %d, %Y}?",
            f"Can I swing by {res_name} on {final_time:%Y/%m/%d} at {t}? Will it be open?",
            f"It's currently {random_format(final_time)}. Is {res_name} still open?"
        ])

    if problem_type == PromptType.TO_BOOL and use_timedelta:
        if time_delta.days == 1:
            return f"Will {res_name} be open tomorrow at {t}?"
        
        if time_delta.days == 2 and r < 0.5:
            return f"Would {res_name} be open if I went there the day after tomorrow at {t}?"
        
        if time_delta.days == 7 and r < 0.5:
            return f"Will {res_name} be open in a week at {t}?"
        
        return np.random.choice([
            f"Can I visit {res_name} in {time_delta.days} days at {t}?", # >= 2 days
            f"Will {res_name} be open {time_delta.days} days from now at {t}?", # >= 2 days AND replace string AND hour
            f"In {time_delta.days} days, I want to dine at {res_name}. Will it be open around {t}?", # >= 2 days
            f"Could you check if {res_name} will be open {time_delta.days} days from now at {t}?", # >= 2 days
        ])
        
    if problem_type == PromptType.TO_HOURS and not use_timedelta:
        return np.random.choice([
            f"What are the opening hours for {res_name} on {final_time:%Y-%m-%d}?",
            f"Can you tell me the opening hours of {res_name} for the date {final_time:%B %d, %Y}?",
            f"Please provide the operating hours for {res_name} on {final_time:%A, %B %d, %Y}.",
            f"I need to know the business hours of {res_name} for {final_time:%m/%d/%Y}.",
            f"What time does {res_name} open and close on {final_time:%m-%d-%Y}?",
            f"On {final_time:%d %B %Y}, what are the opening hours of {res_name}?",
            f"For the date {final_time:%Y-%m-%d}, what are the hours of operation for {res_name}?",
            f"Could you list the opening and closing times of {res_name} on {final_time:%A, %Y-%m-%d}?",
            f"I am looking for the business hours of {res_name} for the date {final_time:%Y/%m/%d}.",
            f"Please tell me the opening times for {res_name} on the date {final_time:%d-%B-%Y}.",
            f"What are the hours of operation at {res_name} on {final_time:%A, %B %d, %Y}?",
            f"On {final_time:%B %d}, what are the working hours for {res_name}?",
            f"For {final_time:%Y-%m-%d}, what are the open and close times of {res_name}?",
            f"What are the hours {res_name} operates on {final_time:%A, %m/%d/%Y}?"
        ])
    
    if problem_type == PromptType.TO_HOURS and use_timedelta:
        if time_delta.days == 1:
            return f"What are the opening hours for {res_name} tomorrow?"
        return np.random.choice([
            f"Can you tell me the opening hours for {res_name} {time_delta.days} days from now?",
            f"Please provide the operating hours for {res_name} {time_delta.days} days later.",
            f"What time does {res_name} open and close on {time_delta.days} days later?",
            f"What are the business hours for {res_name} {time_delta.days} days in the future?",
            f"What are the opening hours for {res_name} {time_delta.days} days later?"
        ])
        
    if problem_type == PromptType.TO_LIST and not use_timedelta:
        return np.random.choice([
            f"The current time is {final_time:%Y-%m-%d %H:%M:%S}. Which restaurants are open right now?",
            f"It's {final_time:%A, %B %d at %I:%M %p} and I'd like to know which restaurants are open.",
            f"If the time is {final_time:%m/%d/%Y %H:%M} right now, can you list the restaurants that are currently open?",
            f"Given that it is currently {final_time:%I:%M %p on %A, %B %d, %Y}, what restaurants remain open?",
            f"At this exact moment, it's {final_time:%H:%M on %d-%m-%Y}. Please list all the open restaurants.",
            f"The exact time currently is {final_time:%I:%M:%S %p, %b %d, %Y}. Which restaurants are open now?",
            f"Can you tell me the open restaurants if the time is now {final_time:%H:%M:%S} on {final_time:%B %d}?",
            f"Tell me the restaurants that are open at {final_time:%H:%M on %d/%m/%Y}.",
            f"It is currently {final_time:%A at %I:%M %p}. Which places are open?",
            f"It's {final_time:%b %d, %Y, %H:%M %p}, what restaurants can I go to?",
            f"It’s {final_time:%H:%M:%S on %A, %B %d, %Y}. Which restaurants should be open now?",
            f"The time is currently {final_time:%d-%m-%Y %H:%M}. Can you list the restaurants that are still open?",
            f"Find the restaurants open at {final_time:%A, %d %b %Y %H:%M:%S}.",
            f"It’s {final_time:%I:%M %p on %Y-%m-%d}. What restaurants are open at this time?"
        ])
        
    if problem_type == PromptType.TO_LIST and use_timedelta:
        if time_delta.days == 1:
            return np.random.choice([
                f"Which restaurants are open tomorrow at {t}?",
                f"Can you list the restaurants available tomorrow at {final_time:%-I:%M%p}?",
                f"Tell me the restaurants operating tomorrow at {final_time:%-I:%M%p}.",
                f"What restaurants will be serving tomorrow at {final_time:%-I:%M%p}?",
                f"Could you show the restaurants open tomorrow at {final_time:%-I:%M%p}?",
                f"Please list out the restaurants open tomorrow at {final_time:%-I:%M%p}.",
                f"Which food spots will be available tomorrow at {final_time:%-I:%M%p}?",
            ])
        return np.random.choice([
            f"Which restaurants are open {time_delta.days} days from now at {t}?",
            f"Can you find out which restaurants are available {time_delta.days} days later at {t}?",
            f"Please list all the dining options open {time_delta.days} days from the current time at {t}.",
            f"Tell me which places to eat are operating {time_delta.days} days in the future at {t}.",
            f"Identify all the restaurants that will be open {time_delta.days} days ahead at {t}.",
            f"What eating establishments will be open {time_delta.days} days from today at {t}?",
            f"Find all open restaurants {time_delta.days} days from now exactly at {t}.",
            f"Which eateries can I visit {time_delta.days} days from the current date at {t}?",
            f"Give me a list of all restaurants that would be open {time_delta.days} days from now at {t}.",
            f"Check which restaurants are up and running {time_delta.days} days from the present at {t}."
        ])
    return f"No prompts for this yet. Type: {problem_type}, Delta: {use_timedelta}"



def random_format(time: datetime) -> str:
    dates = [
        f"{time:%Y-%m-%d}", f"{time:%Y/%m/%d}",
        f"{time:%m-%d-%Y}", f"{time:%m/%d/%Y}",
    ]
    transitions = [' ',]
    
    if np.random.rand() < 0.5:
        dates += [
            f"{time:%B %d, %Y}", f"{time:%b %d, %Y}",
            f"{time:%A %B %d, %Y}", f"{time:%A %b %d, %Y}",
            f"{time:%a %B %d, %Y}", f"{time:%a %b %d, %Y}",
            f"{time:%A}", f"{time:%a}",
        ]
        transitions += [' at ']
    
    return f"{np.random.choice(dates)}{np.random.choice(transitions)}{random_time_format(time)}"

def random_time_format(time: datetime) -> str:
    if time.minute == 0 and np.random.rand() < 0.5:
        return np.random.choice([
            f"{time:%-I%p}",
        ])
        
    return np.random.choice([
        f"{time:%H:%M:%S}", f"{time:%H:%M}",
        f"{time:%I:%M:%S%p}", f"{time:%I:%M%p}",
        f"{time:%-I:%M:%S%p}", f"{time:%-I:%M%p}",
    ])


if __name__ == '__main__':
    for i in range(20):
        print(get_prompt("ToHours", "<Name>", datetime.now(), True, timedelta(days=i%5+1)))