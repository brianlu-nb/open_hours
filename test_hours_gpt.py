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
from hours_debug import Report

USE_GPT = True
DEBUG = True

if USE_GPT:
    load_dotenv()
    API_KEY = os.getenv('OPENAI_API_KEY')
    client = OpenAI()
    MODEL = 'gpt-4o'
else:
    client = InferenceClient(model="http://0.0.0.0:8080")

def query(prompt: dict) -> str:
        
    if prompt['use_delta']:
        user_prompt_template = prompt_templates.templates_timed[prompt['problem_type']]
    else:
        user_prompt_template = prompt_templates.templates[prompt['problem_type']]
    user_prompt = user_prompt_template.format(
            opening_hours=prompt['opening_hours'],
            today=f"{prompt['today']:%B %d, %Y}",
            user_prompt=prompt['user_prompt'],
    )
            
    if USE_GPT:
        response = client.chat.completions.create(
            model=MODEL,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": prompt_templates.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content
    
    prompt = prompt_templates.to_llama_format(prompt_templates.system_prompt, user_prompt)

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
            ptype.TO_BOOL: Report(ptype.TO_BOOL),
            ptype.TO_HOURS: Report(ptype.TO_HOURS),
            ptype.TO_LIST: Report(ptype.TO_LIST)
        }
        debug_reports_timed = {
            ptype.TO_BOOL: Report(ptype.TO_BOOL),
            ptype.TO_HOURS: Report(ptype.TO_HOURS),
            ptype.TO_LIST: Report(ptype.TO_LIST)
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

                with open('open_hours/out/report.out', 'w', encoding='utf-8') as file:
                    file.write(f"{tally.report_str(f'Output as of {datetime.now():%B %d, %Y at %H:%M:%S}')}\n\n")
                    for type in list(ptype):
                        if debug_reports[type].count > 0:
                            file.write(f"{debug_reports[type].report_str(f'----- {type}, False -------')}\n\n")
                        if debug_reports_timed[type].count > 0:
                            file.write(f"{debug_reports_timed[type].report_str(f'----- {type}, True ------')}\n\n")

                with open('open_hours/out/responses.jsonl', 'a', encoding='utf-8') as file:
                    file.write(f'{json.dumps({"id": tally.count, "prompt": prompt.to_dict_presentable(), "full_response": response}, ensure_ascii=False)}\n')

            pbar.set_postfix(tally.postfix())
            pbar.update()

if __name__ == "__main__":
    with open(f'{generate_hours.current_path()}/hours.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        prompt_lists = [generate_prompts(data, min_entries=3, max_entries=5, num_trials=100)]
        for prompts in prompt_lists:
            run_prompts(prompts)