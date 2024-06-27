"""Generate answers with GPT-4

Usage:
python3 get_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures
import logging
import shortuuid
from tqdm import tqdm


from fastchat.llm_judge.common import (
    load_questions,
    temperature_config,
    chat_compeletion_openai,
    chat_compeletion_anthropic,
    chat_compeletion_palm,
    call_nb_service,
    call_gpt_service
)
from fastchat.llm_judge.gen_model_answer import reorg_answer_file
from fastchat.model.model_adapter import get_conversation_template
from fastchat.utils import is_ask_nb_model
from inference.gpt_worker import gpt4_prompt


# def get_answer(question: dict, model: str, num_choices: int, max_tokens: int, answer_file: str):
def get_answer(question, model, num_choices, max_tokens, answer_file):
    if args.force_temperature:
        temperature = args.force_temperature
    elif question.get("category") in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    chat_state = None  # for palm-2 model
    for i in range(num_choices):
        conv = get_conversation_template(model)

        turns = []
        for j in tqdm(range(len(question["turns"]))):
            llama_prompt = question["turns"][j]
            print(f"print conv {llama_prompt}")
            if llama_prompt.find("[INST]") < 0:
                conv.append_message(conv.roles[0], llama_prompt)
                conv.append_message(conv.roles[1], None)
                continue
            while True:
                offset_0 = llama_prompt.find("[INST]")
                if offset_0 < 0:
                    break
                offset_1 = llama_prompt.find("[/INST]", offset_0)
                if offset_1 < 0:
                    break
                q = llama_prompt[offset_0 + len("[INST]"): offset_1].strip()
                conv.append_message(conv.roles[0], q)
                llama_prompt = llama_prompt[offset_1 + len("[/INST]"):]
                offset_0 = llama_prompt.find("[INST]")
                if offset_0 < 0:
                    break
                a = llama_prompt[0: offset_0].strip()
                conv.append_message(conv.roles[1], a)


            if model in ["gpt-3.5-turbo"]:
                output = chat_compeletion_openai(model, conv, temperature, max_tokens)
            elif model == "gpt-4":
                output = ""
                results = []
                completion = gpt4_prompt(conv.to_openai_api_messages(), k=1, stream=True, results=results)
                for token in completion:
                    if isinstance(token, str):
                        output = "gpt4 is busy. Please try again later!"
                        break
                    try:
                        x = token['choices'][0]['delta'].get('content')
                        if x is not None:
                            output += x
                    except:
                        print('get ans error')
                        break

            elif model in ["claude-v1", "claude-instant-v1"]:
                output = chat_compeletion_anthropic(
                    model, conv, temperature, max_tokens
                )
            elif model == "palm-2-chat-bison-001":
                chat_state, output = chat_compeletion_palm(
                    chat_state, model, conv, temperature, max_tokens
                )
            elif is_ask_nb_model(model) or model.find('llama-2') >= 0:
                output = call_nb_service(model, conv, temperature, max_tokens)
            elif model.find('gpt') >= 0:
                output = call_gpt_service(conv.get_prompt())
            else:
                raise ValueError(f"Invalid judge model name: {model}")

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model}.jsonl"
    print(f"Output to {answer_file} question_file {question_file}")
    
    for question in questions:
        print(f"to call get_answer", question, args.model, args.num_choices, args.max_tokens, answer_file)
        get_answer(question, args.model, args.num_choices, args.max_tokens, answer_file)

    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
    #     futures = []
    #     for question in questions:
    #         future = executor.submit(
    #             get_answer,
    #             question,
    #             args.model,
    #             args.num_choices,
    #             args.max_tokens,
    #             answer_file,
    #         )
    #         futures.append(future)

    #     for future in tqdm.tqdm(
    #         concurrent.futures.as_completed(futures), total=len(futures)
    #     ):
    #         future.result()

#    reorg_answer_file(answer_file)
