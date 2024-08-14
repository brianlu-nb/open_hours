from multipledispatch import dispatch
from hours_prompts import PromptType as ptype

template_to_list = (
'''Given a list of weekly restaurant hours and some user inquiry including an exact date and time, determine whether each of the restaurants are open then.
While doing so, consider the following:
1. "time": the actual date and time in the user inquiry
2. "day_of_week": the exact day of the week that "time" would be
3. "opening_hours": the opening hours of the restaurants in "day_of_week"
4. "is_open": whether each of the restaurants are open, "true" if a restaurant is open at the "time" in the "opening_hours"
Below are the data for the opening hours of restaurants, today\'s date, and the user question:

In the response, copy all the names exactly. Note that the durations from the hours are inclusive on the left and exclusive on the right.

Opening Hours: {opening_hours}
Today is {today}
User Question: {user_prompt}
Please output in the following json format:
{{
    "time": "%Y-%m-%d %I:%M%p",
    "day_of_week": "%a",
    "opening_hours": {{
        "Restaurant Name 1": "%I:%M%p-%I:%M%p",
        "Restaurant Name 2": "%I:%M%p-%I:%M%p, %I:%M%p-%I:%M%p",
        "Restaurant Name 3": "Closed",
        ...
    }},
    "is_open": {{
        "Restaurant Name 1": true/false,
        ...
    }}
}}
Please give your response below:
''')

template_to_list_timed = (
'''Given a list of weekly restaurant hours and some user inquiry including some relative date and time, determine whether each of the restaurants are open then.
While doing so, consider the following:
1. "time": the date and time asked for in the user inquiry
2. "day_of_week": the exact day of the week that "time" would be
3. "opening_hours": the opening hours of the restaurants in "day_of_week"
4. "is_open": whether each of the restaurants are open, "true" if a restaurant is open at the "time" in the "opening_hours"
Below are the data for the opening hours of restaurants, today\'s date, and the user question:

In the response, copy all the names exactly. Note that the durations from the hours are inclusive on the left and exclusive on the right.

Opening Hours: {opening_hours}
Today is {today}
User Question: {user_prompt}
Please output in the following json format:
{{
    "time": "%Y-%m-%d %I:%M%p",
    "day_of_week": "%a",
    "opening_hours": {{
        "Restaurant Name 1": "%I:%M%p-%I:%M%p",
        "Restaurant Name 2": "%I:%M%p-%I:%M%p, %I:%M%p-%I:%M%p",
        "Restaurant Name 3": "Closed",
        ...
    }},
    "is_open": {{
        "Restaurant Name 1": true/false,
        ...
    }}
}}
Please give your response below:
''')

template_to_bool = (
'''Given a list of weekly restaurant hours and some user inquiry including a restuarant name and an exact date and time, determine whether the named restaurant is open then.
While doing so, consider the following:
1. "name": the name of the restaurant in the user inquiry
2. "time": the actual date and time in the user inquiry
3. "day_of_week": the exact day of the week that "time" would be
4. "opening_hours": the opening hours of the named restaurant in "day_of_week"
5. "is_open": "true" if a restaurant is open at the "time" in the "opening_hours"
Below are the data for the opening hours of restaurants, today\'s date, and the user question:

In the response, copy all the names exactly. Note that the durations from the hours are inclusive on the left and exclusive on the right.

Opening Hours: {opening_hours}
Today is {today}
User Question: {user_prompt}
Please output in the following json format:
{{
    "name": "Restaurand Name",
    "time": "%Y-%m-%d %I:%M%p",
    "day_of_week": "%a",
    "opening_hours": "%I:%M%p-%I:%M%p"/"%I:%M%p-%I:%M%p, %I:%M%p-%I:%M%p"/"Closed",
    "is_open": true/false
}}
Please give your response below:
''')

template_to_bool_timed = (
'''Given a list of weekly restaurant hours and some user inquiry including a restuarant name and some relative date and time, determine whether the named restaurant is open then.
While doing so, consider the following:
1. "name": the name of the restaurant in the user inquiry
2. "time": the date and time asked for in the user inquiry
3. "day_of_week": the exact day of the week that "time" would be
4. "opening_hours": the opening hours of the named restaurant in "day_of_week"
5. "is_open": "true" if a restaurant is open at the "time" in the "opening_hours"
Below are the data for the opening hours of restaurants, today\'s date, and the user question:

In the response, copy all the names exactly. Note that the durations from the hours are inclusive on the left and exclusive on the right.

Opening Hours: {opening_hours}
Today is {today}
User Question: {user_prompt}
Please output in the following json format:
{{
    "time": "%Y-%m-%d %I:%M%p",
    "day_of_week": "%a",
    "opening_hours": "%I:%M%p-%I:%M%p"/"%I:%M%p-%I:%M%p, %I:%M%p-%I:%M%p"/"Closed",
    "is_open": true/false
}}
Please give your response below:
''')

template_to_hours = (
'''Given a list of weekly restaurant hours and some user inquiry including a restuarant name and an exact date, determine the opening hours the restaurant that day.
While doing so, consider the following:
1. "name": the name of the restaurant in the user inquiry
2. "time": the actual date and time in the user inquiry
3. "day_of_week": the exact day of the week that "time" would be
4. "opening_hours": the opening hours of the named restaurant in "day_of_week"
Below are the data for the opening hours of restaurants, today\'s date, and the user question:

In the response, copy all the names exactly. Note that the durations from the hours are inclusive on the left and exclusive on the right.

Opening Hours: {opening_hours}
Today is {today}
User Question: {user_prompt}
Please output in the following json format:
{{
    "name": "Restaurand Name",
    "time": "%Y-%m-%d 12:00AM",
    "day_of_week": "%a",
    "opening_hours": "%I:%M%p-%I:%M%p"/"%I:%M%p-%I:%M%p, %I:%M%p-%I:%M%p"/"Closed"
}}
Please give your response below:
''')

template_to_hours_timed = (
'''Given a list of weekly restaurant hours and some user inquiry including a restuarant name and some relative date, determine the opening hours the restaurant that day.
While doing so, consider the following:
1. "name": the name of the restaurant in the user inquiry
2. "time": the date and time asked for in the user inquiry
3. "day_of_week": the exact day of the week that "time" would be
4. "opening_hours": the opening hours of the named restaurant in "day_of_week"
Below are the data for the opening hours of restaurants, today\'s date, and the user question:

In the response, copy all the names exactly. Note that the durations from the hours are inclusive on the left and exclusive on the right.

Opening Hours: {opening_hours}
Today is {today}
User Question: {user_prompt}
Please output in the following json format:
{{
    "name": "Restaurand Name",
    "time": "%Y-%m-%d 12:00AM",
    "day_of_week": "%a",
    "opening_hours": "%I:%M%p-%I:%M%p"/"%I:%M%p-%I:%M%p, %I:%M%p-%I:%M%p"/"Closed"
}}
Please give your response below:
''')

system_prompt = "You are a time management assistant performing time related tasks."

templates = {
    ptype.TO_BOOL: template_to_bool,
    ptype.TO_HOURS: template_to_hours,
    ptype.TO_LIST: template_to_list
}

templates_timed = {
    ptype.TO_BOOL: template_to_bool_timed,
    ptype.TO_HOURS: template_to_hours_timed,
    ptype.TO_LIST: template_to_list_timed
}

format_llama = (
'''<|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assisstant<|end_header_id|>

''')

@dispatch(str, str)
def to_llama_format(system_prompt: str, user_prompt: str) -> str:
    return format_llama.format(system_prompt = system_prompt, user_prompt = user_prompt)

@dispatch(dict)
def to_llama_format(prompt: dict) -> str:
    if prompt['use_delta']:
        user_prompt_template = templates_timed[prompt['problem_type']]
    else:
        user_prompt_template = templates[prompt['problem_type']]
    user_prompt = user_prompt_template.format(
            opening_hours=prompt['opening_hours'],
            today=prompt['today'],
            user_prompt=prompt['user_prompt'],
    )  
    return format_llama.format(system_prompt = system_prompt, user_prompt = user_prompt)


if __name__ == '__main__':
    print(template_to_bool.format(opening_hours=1, today=2, user_prompt=3))