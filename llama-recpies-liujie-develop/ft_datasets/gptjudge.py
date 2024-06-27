import json
import os
import dataclasses
import re
import logging
import openai, ast, time, anthropic
from fastchat.model.model_adapter import get_conversation_template
from tqdm import tqdm

TIE_DELTA = 0.1
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
API_MAX_RETRY = 100
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"
judge_dict = {
    "event": "pair-event",
    "entity": "pair-fd",
    "news": "pair-news-search",
    "auto": "pair-auto",
    "hotel": "pair-v2",
    "shopping": "pair-shopping",
}

def chat_compeletion_openai(model, conv, temperature, max_tokens):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                api_base="http://cooper.k8s.nb-prod.com/v1",
                api_key="nb-2YvIn-SBDiIMhNVFDa483SC8h81wbP8z-JvN0tPo3pUi5PVdbIQsowtC2JFead6JXrk"
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def chat_compeletion_anthropic(model, conv, temperature, max_tokens):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            prompt = conv.get_prompt()
            response = c.completions.create(
                model=model,
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
            )
            output = response.completion
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output.strip()



def load_judge_prompts(prompt_file: str):
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = False
    multi_turn: bool = False


@dataclasses.dataclass
class MatchPair:
    question: dict
    model_1: str
    model_2: str
    answer_1: dict
    answer_2: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False


def run_judge_pair(question, answer_a, answer_b, judge, ref_answer, multi_turn=False):
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_a_1=answer_a["choices"][0]["turns"][0],
            answer_b_1=answer_b["choices"][0]["turns"][0],
            answer_a_2=answer_a["choices"][0]["turns"][1],
            answer_b_2=answer_b["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer_a=answer_a["choices"][0]["turns"][0],
            answer_b=answer_b["choices"][0]["turns"][0],
            **kwargs,
        )

    winner = "error"
    if not answer_a["choices"][0]["turns"][0] or not answer_b["choices"][0]["turns"][0]:
        logging.info(f"empty answer. skip")
        return winner, user_prompt, "empty answer. skip"
    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if model in ["gpt-3.5-turbo", "gpt-4"]:
        conv.set_system_message(system_prompt)
        judgment = chat_compeletion_openai(model, conv, temperature=0.1, max_tokens=2048)
    elif model in ["claude-v1", "claude-instant-v1"]:
        if system_prompt != "You are a helpful assistant.":
            user_prompt = "[Instruction]\n" + system_prompt + "\n\n" + user_prompt
            conv.messages[0][1] = user_prompt
        judgment = chat_compeletion_anthropic(
            model, conv, temperature=0.1, max_tokens=1024
        )
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] == "[[A]]":
        if "[[A]]" in judgment:
            winner = "A"
        elif "[[B]]" in judgment:
            winner = "B"
        elif "[[C]]" in judgment:
            winner = "tie"
        else:
            winner = "error"
    elif judge.prompt_template["output_format"] == "[[rating_a,rating_b]]":
        match = re.search(two_score_pattern, judgment)
        if not match:
            match = re.search(two_score_pattern_backup, judgment)
        if match:
            scores = [ast.literal_eval(s.strip()) for s in match.groups()]
            if abs(scores[0] - scores[1]) <= TIE_DELTA:
                winner = "tie"
            elif scores[0] > scores[1]:
                winner = "A"
            else:
                winner = "B"
        else:
            winner = "error"
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return winner, user_prompt, judgment



def play_a_match_pair(match: MatchPair):
    question, model_1, model_2, answer_1, answer_2, judge, ref_answer, multi_turn = (
        match.question,
        match.model_1,
        match.model_2,
        match.answer_1,
        match.answer_2,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )

    g1_winner, g1_user_prompt, g1_judgment = run_judge_pair(
        question, answer_1, answer_2, judge, ref_answer, multi_turn=multi_turn
    )
    g2_winner, g2_user_prompt, g2_judgment = run_judge_pair(
        question, answer_2, answer_1, judge, ref_answer, multi_turn=multi_turn
    )

    g1_map = {"A": "model_1", "B": "model_2"}
    g2_map = {"A": "model_2", "B": "model_1"}
    g1_winner = g1_map.get(g1_winner, g1_winner)
    g2_winner = g2_map.get(g2_winner, g2_winner)
    turn = 1 if not multi_turn else 2

    result = {
        "model_1": model_1,
        "model_2": model_2,
        "g1_winner": g1_winner,
        "g2_winner": g2_winner,
        # "judge": (judge.model_name, judge.prompt_template["name"]),
        # "g1_user_prompt": g1_user_prompt,
        "g1_judgment": g1_judgment,
        # "g2_user_prompt": g2_user_prompt,
        "g2_judgment": g2_judgment,
        # "turn": turn,
        # "tstamp": time.time(),
    }

    print(
        f"turn: {turn}, model_1: {model_1}, model_2: {model_2}, "
        f"g1_winner: {g1_winner}, g2_winner: {g2_winner}, "
        f"judge: {(judge.model_name, judge.prompt_template['name'])}"
    )

    return result


def make_judge_pairwise(judge_model, judge_prompts, judge_prompt='pair-v2'):
    judges = {}
    print(f'using judgement-prompt {judge_prompt}')
    judges["default"] = Judge(judge_model, judge_prompts[judge_prompt])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


judge_prompts = load_judge_prompts('judge_prompts.jsonl')

prefix = "gy_dpo_1211"
questions = []
with open("../data/%s.jsonl"%prefix, "r") as fr:
    lines = fr.readlines()

for l in lines:
    jline = json.loads(l)
    jline['turns'] = [jline['query']]

    questions.append(jline)

data_list = []

with open("../data/%s_judged.jsonl"%prefix, "w") as fw:
    for q in tqdm(questions):

        answer = {"choices": [{"turns": [q["answer"]]}]}
        rejected_answer = {"choices": [{"turns": [q["rejected_response"]]}]}

        type = q["type"]

        judges = make_judge_pairwise('gpt-4', judge_prompts, judge_dict.get(type, "pair-v2"))
        match = MatchPair(q, 'gpt4', 'nbmodel', answer, rejected_answer, judges['default'], multi_turn=False)
        print('Judgment pending...')
        result = play_a_match_pair(match)

        q.pop("turns")

        q.update(result)

        fw.write(json.dumps(q) + "\n")
        fw.flush()
