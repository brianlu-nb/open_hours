import numpy as np
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import hours_prompts
from hours_prompts import Prompt, generate_prompts
from hours_prompts_db import PromptType as ptype
import generate_hours
from huggingface_hub import InferenceClient



def query(prompt: dict, use_openai: bool) -> str:
    if use_openai:
        with open("test_hours/hours_system_prompt.txt", 'r') as file:
            system_prompt = file.read()
        with open("test_hours/hours_user_prompt.txt", 'r') as file:
            user_prompt = file.read()
            # print(prompt)
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
    
    with open("test_hours/prompt_without_reason.txt", 'r') as file:
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

def print_report(title: str, tp: int, fp: int, fn: int, tn: int, nfp: int, nfn: int, nkp: int, nkn: int, err: int):
    total = tp + tn + fp + fn + nfp + nfn
    ep = tp + fn + nfp
    en = fp + tn + nfn
    rp = tp + fp + nkp
    rn = tn + fn + nkn
    rnf = nfp + nfn
    tnk = nkp + nkn
    pprec = -1 if rp == 0 else tp / rp * 100
    precl = -1 if ep == 0 else tp / ep * 100
    nprec = -1 if rn == 0 else tn / rn * 100
    nrecl = -1 if en == 0 else tn / en * 100
    pf1 = 2/(1/(pprec+1e-10)+1/(precl+1e-10))
    nf1 = 2/(1/(nprec+1e-10)+1/(nrecl+1e-10))
    acc = -1 if total == 0 else (tp + tn) / total * 100

    out = ''
    with open("test_hours/report_template.txt", 'r') as file:
        out = file.read().format(
            title=title,
            tp=tp, tn=tn, fp=fp, fn=fn, total=total,
            ep=ep, en=en, rp=rp, rn=rn,
            nfp=nfp, nfn=nfn, rnf=rnf,
            pprec=pprec, nprec=nprec, precl=precl, nrecl=nrecl,
            pf1=pf1, nf1=nf1,
            nkp=nkp, nkn=nkn, tnk=tnk,
            acc=acc, err=err,
        )

    with open("test_hours/_report.out", 'w') as file:
        file.write(out)

def evaluate(id: int, response: str, prompt: Prompt, correct_ds: list, incorrect_ds: list) -> tuple[float, float, float, float, float]:
    if isinstance(prompt, dict):
        prompt = Prompt.wrap_dict(prompt)
    with open('test_hours/response-2.out', 'w') as file:
        file.write(response)
    
    res_str = response
    correct = 0
    tp, fp, fn, tn, nfp, nfn = 0, 0, 0, 0, 0, 0
    nkp, nkn = 0, 0
    num_errors = 0
    try:
        response_dict = json.loads(response)
        # correct = prompt.evaluate_response(response_dict)
        # print(f"prompt: {user_prompt}")
        # print(f"response: {response}")
        # print(f"{is_correct}")
        res_str = response_dict
        tp, fp, fn, tn, nfp, nfn = prompt.output_report(response_dict)
        nkp, nkn = prompt.new_keys(response_dict)
        correct = (tp + tn) / sum([tp, fp, fn, tn, nfp, nfn])
        # print(tp + tn, sum([tp, fp, fn, tn, nfp, nfn]), correct)
    except:
        num_errors = 1
        
    
    precision = tp / (tp + fp + nkp) if (tp + fp + nkp) != 0 else ''
    recall = tp / (tp + fn + nfp) if (tp + fn + nfp) != 0 else ''
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
    return correct, tp, fp, fn, tn, nfp, nfn, nkp, nkn, num_errors

def report(num_trials: int, num_correct: int, correct_ds: list, incorrect_ds: list):
    with open(f"test_hours/_correct.json", 'w') as f_correct:
        json.dump(correct_ds, f_correct, indent=4)
    
    with open(f"test_hours/_incorrect.json", 'w') as f_incorrect:
        json.dump(incorrect_ds, f_incorrect, indent=4)
    
    with open(f'test_hours/_hours.out', 'a') as file:
        file.write(f'----------------{num_correct}/{num_trials}----------------\n\n')
        
def run_one_prompt(prompt_type: ptype, use_delta: bool, num_trials: int, num_completed: int):
    print(f'Type: {prompt_type}, Delta: {use_delta}')
    correct_ds = []
    with open(f"test_hours/_correct.json", 'r') as file:
        correct_ds = json.load(file)
    
    incorrect_ds = []
    with open(f"test_hours/_incorrect.json", 'r') as file:
        incorrect_ds = json.load(file)

    # prompt_list = generate_prompts(hours_dict, prompt_type, use_delta, num_trials)
    # all_prompts = generate_hours.generate_data(prompt_type, use_delta, num_trials, (0.8, 0.1, 0.1))
    
    with open(f"{generate_hours.current_path()}/prompts_eval.json", 'r') as file:
        prompt_list = json.load(file)
    
    # for prompts, split in zip(all_prompts, ("train", "test", "eval")):
    num_correct = 0
    num_accurate = 0
    true_positives = 0
    reported_positives = 0
    expected_positives = 0
    tot_tp, tot_fp, tot_fn, tot_tn, tot_nfp, tot_nfn = 0, 0, 0, 0, 0, 0
    tot_nkp, tot_nkn = 0, 0
    num_errors = 0

    with tqdm(desc=f"Querying", total=len(prompt_list)) as t:
    # with tqdm(desc=f"Querying {split}", total=len(prompts)) as t:
    #     with open(f'{generate_hours.current_path()}/{split}.json', 'w') as file:
    #         file.write("[]")
        for i in range(len(prompt_list)):
            prompt = prompt_list[i]
            # prompt = prompts[i]
            # response = query(prompt.to_dict(), False)
            response = query(prompt, False)
            
            for _ in range(10):
                accuracy, tp, fp, fn, tn, nfp, nfn, nkp, nkn, num_err = evaluate(i + num_completed, response, prompt, correct_ds, incorrect_ds)
                num_errors += num_err * 0.1
                if num_err == 0:
                    break
            
            num_accurate += accuracy
            
            if accuracy == 1:
                num_correct += 1
                # with open(f'{generate_hours.current_path()}/{split}.json', 'r') as file:
                #     recent = json.load(file)
                # with open(f'{generate_hours.current_path()}/{split}.json', 'w') as file:
                #     recent.append(correct_ds[-1])
                #     json.dump(recent, file, indent=4)  

            true_positives += tp
            reported_positives += tp + fp
            expected_positives += tp + fn
            
            tot_tp += tp
            tot_fp += fp
            tot_fn += fn
            tot_tn += tn
            tot_nfp += nfp
            tot_nfn += nfn
            tot_nkp += nkp
            tot_nkn += nkn
            print_report("Output", tot_tp, tot_fp, tot_fn, tot_tn, tot_nfp, tot_nfn, tot_nkp, tot_nkn, num_errors)

            if (i + 1) % 10 == 0:
                report(i + 1, num_correct, correct_ds, incorrect_ds)

            t.postfix = (
                f"# Correct: {num_correct}, "
                f"# Error: {num_errors:.0f}, "
                f"Accuracy: {((tot_tp + tot_tn + 1e-10) / (sum([tot_tp, tot_fp, tot_fn, tot_tn]) + 1e-10) * 100):.2f}%, "
                f"Precision: {((tot_tp + 1e-10) / (tot_tp + tot_fp + 1e-10) * 100):.2f}%, "
                f"Recall: {((tot_tp + 1e-10) / (tot_tp + tot_fn + 1e-10) * 100):.2f}%"
            )
            t.update()

    print()
    report(num_trials, num_correct, correct_ds, incorrect_ds)

# runs

MODEL = "gpt-4o"
TEMPERATURE = 1.0

# load_dotenv()
# api_key = os.getenv('OPENAI_API_KEY')
# client = OpenAI()
client = InferenceClient(model="http://0.0.0.0:8080")

with open(f"test_hours/_correct.json", 'w') as file:
    file.write('[]')
with open(f"test_hours/_incorrect.json", 'w') as file:
    file.write('[]')
with open(f"test_hours/_hours.out", 'w') as file:
    file.write('')

with open(f"{generate_hours.current_path()}/hours.json", 'r') as file:
    hours_dict = json.load(file)
run_one_prompt(ptype.TO_LIST, False, 1000, 0)

# prompts = generate_prompts(prompt_type=ptype.TO_LIST, use_delta=False, num_trials=10000)
# with open("test_hours/data-2.jsonl", 'w') as file:
#     file.writelines([f"{json.dumps(prompt.to_dict_presentable())}\n" for prompt in prompts])