import copy
import json
import logging

import os
import random
import sys

# append file path
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
print(FILE_PATH)
sys.path.append(FILE_PATH + "/")

import torch
from huggingface_hub import InferenceClient
from transformers import (
    TextGenerationPipeline
)
from transformers.models.llama import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer
from rouge import Rouge
from pydantic import BaseModel, conint
import requests
from text_generation import Client
from ft_datasets import unified_dataset, prompt_util
import datetime
from multiprocessing import Process, Queue
import tqdm
from queue import Empty
import argparse
from sklearn import metrics
from ft_datasets import finetune_prompt


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ["WORLD_SIZE"] = "1"

B_INST, E_INST = "[INST]", "[/INST]"
B_SENTENCE, E_SENTENCE = "<s>", "</s>"


class ContentQualityResult(BaseModel):
    Clarity_and_Coherence_Score: conint(ge=1, le=5)
    Clarity_and_Coherence_Evaluation_Rationale: str
    Relevance_and_Focus_Score: conint(ge=1, le=5)
    Relevance_and_Focus_Evaluation_Rationale: str
    Originality_and_Insightfulness_Score: conint(ge=1, le=5)
    Originality_and_Insightfulness_Evaluation_Rationale: str
    Engagement_and_Interest_Score: conint(ge=1, le=5)
    Engagement_and_Interest_Evaluation_Rationale: str
    Grammar_and_Style_Score: conint(ge=1, le=5)
    Grammar_and_Style_Evaluation_Rationale: str
    Structural_Integrity_Score: conint(ge=1, le=5)
    Structural_Integrity_Evaluation_Rationale: str
    Argumentation_and_Evidence_Score: conint(ge=1, le=5)
    Argumentation_and_Evidence_Evaluation_Rationale: str
    Overall_Score: conint(ge=1, le=5)
    Overall_Evaluation_Rationale: str

class LLamaInfer(object):

    def __init__(self):
        # self.model_path = "/mnt/share16t/liujie/code_workspace/llama-recpies/exp/llama_13bf_finetune_v1"
        self.model_path = "/mnt/share16t/liujie/code_workspace/llama-recpies/exp/llama_13bf_finetune_v1"
        # self.model_path = "meta-llama/Llama-2-13b-hf"
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     pretrained_model_name_or_path=self.model_path,
        #     device_map="cuda",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2"
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        self.model = LlamaForCausalLM.from_pretrained(self.model_path, device_map=torch.device("cuda:1"))
        self.model.eval()

        self.tokenizer.model_max_length = 4096
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token_id = 0
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer, framework="pt")

    def infer(self, prompt: str):
        output = self.pipeline(prompt, do_sample=False)
        return output


class TGIInfer(object):

    def __init__(self, multi=False, port_in_use=8311):
        self.multi = multi
        if self.multi:
            ports = ["8311", "8312", "8313", "8314", "8315", "8316"]
            self.client_list = []
            for port in ports:
                url = "http://127.0.0.1:" + port
                self.client_list.append(InferenceClient(model=url))
        else:
            self.client = InferenceClient(model="http://127.0.0.1:" + str(port_in_use))
            self.text_generation_client = Client(base_url="http://127.0.0.1:" + str(port_in_use))

    def infer(self, prompt):
        client = random.sample(self.client_list, 1)[0] if self.multi else self.client
        output = client.text_generation(prompt=prompt, max_new_tokens=1024)
        return output

    def infer_use_http(self, prompt):
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024
            }
        }
        headers = {
            "Content-Type": "application/json",
        }

        response = requests.post(
            'http://nb-llm-multi-dim-quality.k8s.nb-prod.com/generate',
            headers=headers,
            json=data
        )
        response_json = response.json()
        return response_json["generated_text"] if "generated_text" in response_json else {}

    def infer_use_http_multi_dim_server(self, title, content):
        data = {
            "seg_title": title,
            "seg_content": content
        }
        headers = {
            "Content-Type": "application/json",
        }
        remote_url = "http://multi-dim-quality-server.default.svc.k8sc1.nb.com:8080/writing_quality"


        response = requests.post(
            url=remote_url,
            headers=headers,
            json=data
        )
        return response.json()

    def infer_with_format(self, prompt):
        data = {
            "inputs": prompt,
            "parameters": {
                "grammar": {
                    "type": "json",
                    "value": finetune_prompt.content_quality_json_schema
                }
            }
        }

        headers = {
            "Content-Type": "application/json",
        }

        response = requests.post(
            'http://127.0.0.1:8311/generate',
            headers=headers,
            json=data
        )
        response_json = response.json()
        return response_json["generated_text"] if "generated_text" in response_json else {}


class InferSpeed(object):

    def __init__(self):
        pass

    def call_client(self, i, model, prompts, queue, is_bar=False):
        logger.info(f"process {i}")

        infer_obj = None
        if model == "tgi":
            infer_obj = TGIInfer(port_in_use=8311)
        elif model == "tensorrt":
            infer_obj = TensorRTLLMInfer()
        else:
            raise ValueError(f"{model} is not supported")

        pbar = tqdm.tqdm(total=len(prompts)) if is_bar else None
        for prompt in prompts:
            if is_bar:
                pbar.update(1)
            start = datetime.datetime.now()
            answer = infer_obj.infer(prompt)
            # answer = infer_obj.infer_use_http(prompt)
            # answer = infer_obj.infer_use_http_multi_dim_server(title=prompt.get("title", ""), content=prompt.get("content", ""))
            print(answer)
            end = datetime.datetime.now()
            time_deta = end - start
            time_ms = time_deta.seconds * 1000 + time_deta.microseconds / 1000
            print(f"cost time: {time_ms} ms")
            queue.put({
                "answer": answer,
                "cost_time": time_ms
            })
        if is_bar:
            pbar.close()

    def gather_result(self, i, queue):
        res_list = []
        while True:
            try:
                one_result = queue.get(timeout=60)
                res_list.append(one_result)
            except Empty as e:
                break
        cost_time = [res["cost_time"] for res in res_list]
        max_time = max(cost_time)
        min_time = min(cost_time)
        avg_time = sum(cost_time) / len(cost_time)

        print(f"res len: {len(res_list)}, max_time: {max_time} , min_time: {min_time} , avg_time: {avg_time}")

    def run_infer_speed(self, model, parallel_n, stride=2):
        print(f"##############{model} {parallel_n} begin#########################")
        prompts = []
        # input_file = "/mnt/share16t/liujie/code_workspace/dataset/content_quality_dataset/test.jsonl"
        input_file = "/mnt/nlp/liujie/python_code_workspace/NB-LLM/data_preparation/content_quality/data/test.jsonl"
        with open(input_file, "r", encoding="utf-8") as f:
            for one_line in f:
                one_json = json.loads(one_line.strip())
                # prompt = one_json["prompt"]
                title = one_json.get("title", "")
                content = one_json.get("content", "")
                prompt = _writing_quality_format_prompt(prompt=finetune_prompt.writing_quality_without_rationale_prompt, title=title, content=content)

                prompts.append(prompt)
                # warning temp use json instead of prompt
                # prompts.append(one_json)

        prompts = prompts[: stride * parallel_n]
        lines_len = len(prompts)
        processors = []
        queue = Queue()
        for i in range(parallel_n):
            start = i * stride
            end = (i + 1) * stride
            prompts_i = prompts[start: end]
            is_bar = True if i == parallel_n - 1 else False
            p = Process(target=self.call_client, args=(i, model, prompts_i, queue, is_bar))
            p.start()
            processors.append(p)

        p = Process(target=self.gather_result, args=(parallel_n, queue))
        p.start()
        processors.append(p)
        for one in processors:
            one.join()
        print(f"##############{model} {parallel_n} end#########################")

class TensorRTLLMInfer(object):

    def __init__(self):
        self.url = "http://127.0.0.1:8322/v2/models/tensorrt_llm_bls/generate"

    def infer(self, prompt):
        data = {
            "text_input": prompt,
            "max_tokens": 1024
        }

        headers = {
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.url,
            headers=headers,
            json=data
        )
        response = response.json()
        return response["text_output"]

def _writing_quality_format_prompt(prompt, title, content):
    title_split = title.split(" ")
    title = " ".join(title_split[:256])
    content_split = content.split(" ")
    content = " ".join(content_split[:1536])
    return prompt.format(title=title, content=content)

class MultiProcessInfer(object):

    def __init__(self):
        pass

    def call_client(self, i, model, json_list, queue, is_bar=False):
        logger.info(f"process {i}")

        infer_obj = None
        if model == "tgi":
            infer_obj = TGIInfer(multi=True)
        elif model == "tensorrt":
            infer_obj = TensorRTLLMInfer()
        else:
            raise ValueError(f"{model} is not supported")

        pbar = tqdm.tqdm(total=len(json_list)) if is_bar else None
        for one_json in json_list:
            if is_bar:
                pbar.update(1)
            title = one_json.get("seg_title", "")
            content = one_json.get("seg_content", "")
            if (len(title) == 0 or len(content) == 0):
                continue

            prompt = _writing_quality_format_prompt(prompt=finetune_prompt.writing_quality_without_rationale_prompt, title=title, content=content)

            start = datetime.datetime.now()
            try:
                answer = infer_obj.infer(prompt)
            except Exception as e:
                logger.warning(f"Error in inference: {e}")
                continue
            end = datetime.datetime.now()
            time_deta = end - start
            time_ms = time_deta.seconds * 1000 + time_deta.microseconds / 1000
            one_json["llama_result"] = answer
            # one_json["cost_time"] = time_ms
            queue.put(one_json)
        if is_bar:
            pbar.close()

    def gather_result(self, i, queue, f_out):
        while True:
            try:
                one_result = queue.get(timeout=60)
                f_out.write(json.dumps(one_result) + "\n")
                f_out.flush()
            except Empty as e:
                break

    def _valid(self, one_json, media_id_set):
        title = one_json["stitle"] if "stitle" in one_json else ""
        content = one_json["seg_content"] if "seg_content" in one_json else ""
        content_split = content.split(" ")
        content = " ".join(content_split[:2048])
        score = one_json["score"] if "score" in one_json else 0
        ctype = one_json["ctype"] if "ctype" in one_json else ""
        media_id = one_json["media_id"] if "media_id" in one_json else 0
        if (len(title) == 0 or len(content) == 0 or score <= 0 or len(content_split) < 50 or
                ctype == "native_video" or media_id <= 0 or media_id not in media_id_set):
            return False
        else:
            return True


    def run_infer_multi_process(self, model, parallel_n, input_file, output_file):
        print(f"##############{model} {parallel_n} begin#########################")
        all_json_list = []
        # input_file = "/mnt/share16t/liujie/code_workspace/dataset/creator_sample/detrimental_score_sample.jsonl"
        # output_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/detrimental_sample_result.jsonl"
        # input_file = "/mnt/share16t/liujie/code_workspace/dataset/creator_sample/all_json_list.jsonl"
        # output_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/creator_sample_result.jsonl"
        # input_file = "/mnt/share16t/liujie/code_workspace/dataset/creator_sample/creator_network_review_tasks_enrich_static_feature_score.jsonl"
        # output_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/creator_network_review_tasks_enrich_writing_quality.jsonl"
        # input_file = "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240401.low_quality.jsonl"
        # output_file = "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240401.low_quality.writing_quality.jsonl"

        # media_id_file = "/mnt/share16t/liujie/code_workspace/dataset/creator_sample/media_id_score_filtered.jsonl"
        # media_id_set = set([json.loads(line.strip())["media_id"] for line in open(media_id_file, 'r', encoding='utf-8').readlines()])

        f_out = open(output_file, "w", encoding="utf-8")
        with open(input_file, "r", encoding="utf-8") as f:
            for one_line in f:
                one_json = json.loads(one_line.strip())
                all_json_list.append(one_json)

                # if self._valid(one_json, media_id_set):
                #     all_json_list.append(one_json)
        # for detrimental sample only
        # all_json_list = sorted(all_json_list, key=lambda k: k['detrimental_score'], reverse=True)
        # all_json_list = all_json_list[250: 500]

        score_dict = {}
        for i in range(parallel_n):
            score_dict[i] = []
        for i in range(len(all_json_list)):
            key = i % parallel_n
            score_dict[key].append(all_json_list[i])

        processors = []
        queue = Queue()
        for i in range(parallel_n):
            json_list_i = score_dict[i]
            is_bar = True if i == parallel_n - 1 else False
            p = Process(target=self.call_client, args=(i, model, json_list_i, queue, is_bar))
            p.start()
            processors.append(p)

        p = Process(target=self.gather_result, args=(parallel_n, queue, f_out))
        p.start()
        processors.append(p)
        for one in processors:
            one.join()
        print(f"##############{model} {parallel_n} end#########################")



def test_one():
    prompt = """We measure an article's writing quality based on the following \nGiven a article title and content below, please evaluate its writing quality based on the following 8 dimensions:\n1. Clarity and Coherence: The ability of the article to communicate ideas in a clear and logical manner. This includes the use of straightforward language, well-structured sentences, and a coherent flow of ideas.\n2. Relevance and Focus: The article should stay focused on the central topic or argument without diverging into unrelated areas. Relevance ensures that all parts of the article contribute directly to its main purpose or thesis.\n3. Accuracy and Credibility: Factual accuracy is crucial for building trust with readers. This involves verifying facts, citing reliable sources, and providing accurate representations of data and events.\n4. Originality and Insightfulness: Original content that provides new insights, perspectives, or findings adds significant value. This dimension assesses the uniqueness of the article and its contribution to the subject matter.\n5. Engagement and Interest: The ability to capture and hold the reader's attention through compelling writing, interesting anecdotes, or thought-provoking questions. Engaging writing often includes a strong introduction, dynamic content, and a memorable conclusion.\n6. Grammar and Style: Proper grammar, punctuation, and spelling are fundamental to writing quality. Style also plays a role, including the choice of words, tone, and the use of literary devices that enhance the reading experience.\n7. Structural Integrity: The organization of the article, including the use of headings, subheadings, paragraphs, and bullet points, which can help in making the content more digestible and easier to follow.\n8. Argumentation and Evidence: For articles that aim to persuade or present an argument, the quality of reasoning, the strength of evidence provided, and the effectiveness of the argumentative structure are important. \nBelow is an article title and content:\nTitle: Miami Gardens Police Sergeant Arrested for DUI and Reckless Driving\nContent: A Miami Gardens police sergeant faces charges of driving under the influence (DUI) and reckless driving following an early morning arrest, as reporting WPLG Local10 News. On Thursday morning, authorities confirmed the arrest of 38-year-old Sergeant Andrea Smith by officers from her own department. According to reports, an officer spotted a white Mercedes-Benz driven by Smith speeding erratically and disregarding traffic signals. The police report stated that the car proceeded to roll slowly through a red light, then accelerated, maintaining a high speed while passing through another red light. Officers then switched on the sirens before Smith's car was eventually stopped at approximately 1:50 a.m. near Northwest Second Avenue and 202nd Terrace. Following the traffic stop, the officer observed signs of possible intoxication, including bloodshot eyes, slurred speech, and the odor of alcohol, a police report added. A field sobriety test and Breathalyzer were offered, but Smith reportedly refused. Subsequently, she was arrested for DUI and reckless driving. \"My officers are held to the highest standard both on and off duty\" stated Miami Gardens Police Chief Delma Noel-Pratt. \"I can assure you that the actions of this individual is not the reflection of the agency. The sergeant will be relieved of duty and afforded due process accordingly,\" added Chief Noel-Pratt. As of Thursday morning, Smith remained in custody at the Turner Guilford Knight Correctional Center on a $1,150 bond.\nPlease evaluate its write quality, assign a score (1 to 5) for each dimension, and generate a rationale.\nThen generate a overall quality score (1 to 5) and a rationale.\nPlease output with the following json format \n{\"Clarity_and_Coherence_Score\": score (1 to 5), \"Clarity_and_Coherence_Evaluation_Rationale\": XXX, \n\"Relevance_and_Focus_Score\": score (1 to 5), \"Relevance_and_Focus_Evaluation_Rationale\": XXX, \n\"Accuracy_and_Credibility_Score\": score (1 to 5), \"Accuracy_and_Credibility_Evaluation_Rationale\": XXX, \n\"Originality_and_Insightfulness_Score\": score (1 to 5), \"Originality_and_Insightfulness_Evaluation_Rationale\": XXX, \n\"Engagement_and_Interest_Score\": score (1 to 5), \"Engagement_and_Interest_Evaluation_Rationale\": XXX, \n\"Grammar_and_Style_Score\": score (1 to 5), \"Grammar_and_Style_Evaluation_Rationale\": XXX, \n\"Structural_Integrity_Score\": score (1 to 5), \"Structural_Integrity_Evaluation_Rationale\": XXX, \n\"Argumentation_and_Evidence_Score\": score (1 to 5), \"Argumentation_and_Evidence_Evaluation_Rationale\": XXX, \n\"Overall_Score\": score (1 to 5), \"Overall_Evaluation_Rationale\": XXX}\nPlease output now:\n"""
    without_rational_prompt = '''We measure an article's writing quality based on the following 
Given a article title and content below, please evaluate its writing quality based on the following 8 dimensions:
1. Clarity and Coherence: The ability of the article to communicate ideas in a clear and logical manner. This includes the use of straightforward language, well-structured sentences, and a coherent flow of ideas.
2. Relevance and Focus: The article should stay focused on the central topic or argument without diverging into unrelated areas. Relevance ensures that all parts of the article contribute directly to its main purpose or thesis.
3. Accuracy and Credibility: Factual accuracy is crucial for building trust with readers. This involves verifying facts, citing reliable sources, and providing accurate representations of data and events.
4. Originality and Insightfulness: Original content that provides new insights, perspectives, or findings adds significant value. This dimension assesses the uniqueness of the article and its contribution to the subject matter.
5. Engagement and Interest: The ability to capture and hold the reader's attention through compelling writing, interesting anecdotes, or thought-provoking questions. Engaging writing often includes a strong introduction, dynamic content, and a memorable conclusion.
6. Grammar and Style: Proper grammar, punctuation, and spelling are fundamental to writing quality. Style also plays a role, including the choice of words, tone, and the use of literary devices that enhance the reading experience.
7. Structural Integrity: The organization of the article, including the use of headings, subheadings, paragraphs, and bullet points, which can help in making the content more digestible and easier to follow.
8. Argumentation and Evidence: For articles that aim to persuade or present an argument, the quality of reasoning, the strength of evidence provided, and the effectiveness of the argumentative structure are important. 
Below is an article title and content:
Title: Miami Gardens Police Sergeant Arrested for DUI and Reckless Driving
Content: A Miami Gardens police sergeant faces charges of driving under the influence (DUI) and reckless driving following an early morning arrest, as reporting WPLG Local10 News. On Thursday morning, authorities confirmed the arrest of 38-year-old Sergeant Andrea Smith by officers from her own department. According to reports, an officer spotted a white Mercedes-Benz driven by Smith speeding erratically and disregarding traffic signals. The police report stated that the car proceeded to roll slowly through a red light, then accelerated, maintaining a high speed while passing through another red light. Officers then switched on the sirens before Smith's car was eventually stopped at approximately 1:50 a.m. near Northwest Second Avenue and 202nd Terrace. Following the traffic stop, the officer observed signs of possible intoxication, including bloodshot eyes, slurred speech, and the odor of alcohol, a police report added. A field sobriety test and Breathalyzer were offered, but Smith reportedly refused. Subsequently, she was arrested for DUI and reckless driving. \"My officers are held to the highest standard both on and off duty\" stated Miami Gardens Police Chief Delma Noel-Pratt. \"I can assure you that the actions of this individual is not the reflection of the agency. The sergeant will be relieved of duty and afforded due process accordingly,\" added Chief Noel-Pratt. As of Thursday morning, Smith remained in custody at the Turner Guilford Knight Correctional Center on a $1,150 bond.
Please evaluate its write quality, assign a score (1 to 5) for each dimension.
Then generate a overall quality score (1 to 5).
Please output with the following json format 
{{"Clarity_and_Coherence_Score": score (1 to 5), 
"Relevance_and_Focus_Score": score (1 to 5), 
"Accuracy_and_Credibility_Score": score (1 to 5), 
"Originality_and_Insightfulness_Score": score (1 to 5), 
"Engagement_and_Interest_Score": score (1 to 5), 
"Grammar_and_Style_Score": score (1 to 5), 
"Structural_Integrity_Score": score (1 to 5), 
"Argumentation_and_Evidence_Score": score (1 to 5), 
"Overall_Score": score (1 to 5)}}
Please output now:
'''

    # title = "Insider Today : Google 's big shakeup"
    # content = '''Google CEO Sundar Pichai Mateusz Wlodarczyk/NurPhoto via Getty Images ; Chelsea Jia Feng This post originally appeared in the Insider Today newsletter . You can sign up for Business Insider 's daily newsletter here . Welcome back to our Sunday edition , a roundup of some of our top stories . If you 're interviewing for a job in the coming weeks , check out these five questions to ask at the end . They could help you seal the deal . On the agenda : WiFi Money promised the easy life . Investors wound up ruined . Blue-collar jobs are booming right now . The man leading the fight to legalize MDMA therapy . Inside Google CEO Sundar Pichai 's big leadership shakeup . But first : Markets are throwing off déjà vu vibes . If this was forwarded to you , sign up here . Download Insider 's app here . Roaring Kitty ; Getty Images ; Alyssa Powell/BI This week 's dispatch Investors go back to the future Stocks hit record highs . Roaring Kitty is moving markets . Crypto adverts are on TV . It 's like the past two years never happened . The Dow closed above 40,000 for the first time on Friday , driven by signs of slowing inflation . Keith Gill , the retail trading personality known as Roaring Kitty , tweeted for the first time in three years . The tweet did n't say much , but it was enough to send GameStop 's shares spiking . And with bitcoin hovering near record highs and the crypto winter seemingly behind us , crypto companies are on the front foot once more . The market euphoria stands in stark contrast to what 's going on away from Wall Street . War rages on in Ukraine and the Middle East . Consumers still feel downbeat about the economy . We 're headed for what 's likely to be a deeply contentious US election . For now , investors are shrugging off those concerns . Brandon Celi for BI WiFi Money 's risky allure WiFi Money promises its followers `` the ability to make money anywhere in the world , by doing one simple action ... . connecting to WiFi . '' It sold Americans the idea that with a bit of hustle , they could soon live the easy life . But since its founding in 2020 , the company has left a trail of lawsuits alleging fraud , bankruptcies , mental breakdowns , and financial devastation . Meet the ultimate Instagram hustlers . Erdinhasdemir/Getty Images , Choness/Getty Images , Abanti Chowdhury/BI The hottest jobs right now are blue Want a six-figure salary ? You do n't need to go to Wall Street or Silicon Valley to land one . Instead , blue-collar jobs are booming right now . Jobs that do n't require sitting in front of a computer are in high demand . Demand is high , opportunities abound , and companies like Walmart and UPS offer six-figure salaries and lucrative benefits . It 's getting hot under the blue collar . Also read : Meet the HIFIs : High-income earners who feel financially insecure but ca n't quit their luxury spending AI analyst says plumbers ' and electricians ' jobs are safe but AI models like GPT-4o ` will impact any job that has data ' In a white-collar recession where $ 100,000 jobs are hard to find , women are winning out Richard A. Chance for BI One man 's mission to legalize MDMA For decades , Rick Doblin and MAPS , the nonprofit he founded , have been pushing to legalize medical MDMA . Now , the FDA could issue its approval as soon as this summer . However , insiders have begun voicing concerns about possible ethical lapses in clinical trials and questioning whether MAPS can effectively lead the movement into the future . Why some advocates are sounding the alarm . Google CEO Sundar Pichai Mateusz Wlodarczyk/NurPhoto via Getty Images ; Chelsea Jia Feng Google 's leadership shakeup In April , Google CEO Sundar Pichai executed one of the most dramatic executive shakeups in the company 's 25-year history . Pichai combined two of the company 's major units -- platforms and devices -- into a supergroup focusing on Android , Chrome , and gadgets , like the Pixel phone . How Pichai redesigned his leadership team . Also read : Google just gave us a tantalizing glimpse into the future of AI agents Google 's Android chief says AI will reset the battle with Apple for mobile supremacy This week 's quote : `` They 're just three years late giving the right person the job honestly . '' A former Amazon executive on AWS boss Adam Selipsky 's successor . More of this week 's top reads : High prices have risen from the dead -- and some economists are scared witless . I left JPMorgan and escaped the `` golden handcuffs . '' I have no regrets . Apple needs a revolutionary new product . Is Tim Cook up to the job ? A millennial investor feels happier after ditching most of her Zoom meetings . AI bots are battling it out over job searches . It 's total chaos . Peloton 's demise is another in a long line of failed fitness fad . Adam Neumann is now in the publishing business . A Trump voter proved Mike Lindell wrong . Now he wants his $ 5 million . Heads-up , Apple and the rest of Big Tech : Disney wants better app store deals . The Insider Today team : Matt Turner , deputy editor-in-chief , in New York . Jordan Parker Erb , editor , in New York . Dan DeFrancesco , deputy editor and anchor , in New York . Lisa Ryan , executive editor , in New York . Read the original article on Business Insider'''

    title = "Donald ` I have no idea who Ashton Kutcher is ' Trump continues to make us wish we were being politically punk 'd"
    content = '''There 's a surprising amount of evidence behind the theory that Donald Trump is unable to read , but in reality , literacy is one of few incredibly low bars that he does manage to get over . That does n't mean he 's good at reading , though . The former Apprentice host and current plaintiff in a number of legal cases involving crimes ranging from campaign finance violations to run of the mill business fraud , famously hated having to read while he was president . This might go some way to explaining why he was so incompetent . After leaving court today , Trump convincingly read aloud from a prepared statement in which he compared his legal troubles to the famous early 2000s hit television show Punk 'd : `` This case is a joke . Ashton Kutcher is going to pop and left -LSB- sic -RSB- to tell us we 've been punk 'd ... I think I know what that means . '' Donald Trump : `` `` Ashton Kutcher is going to pop and left to tell us that we 've been punk 'd . '' What ? pic.twitter.com/7uk9OfBES5 -- Republicans against Trump -LRB- @RpsAgainstTrump -RRB- May 16 , 2024 Leaving aside the fact that Trump clearly misread what was written on the page -LRB- or was n't able to ... -RRB- , it 's clear the team around him is devoid of ideas . While Punk 'd was a huge cultural phenomenon in its heyday , nowadays most people would consider it a dated reference . Then again , a large chunk of Trump 's voter base has n't been able to get over the fact the Confederacy lost the Civil War a century and a half ago , so early-noughties MTV is probably quite a modern concept to them . Kutcher is also hardly an underground name , either . Aside from Punk 'd , he was also in the incredibly popular That 70 's Show , and cult classic films like Dude Where 's My Car and The Butterfly Effect . In the 2010s he was even considered to have enough star power to replace Charlie Sheen on , which at the time was one of the biggest shows on network television . Nowadays , Kutcher is probably best known for his advocacy work in ending child sex trafficking via the organization Thorn , which he founded with his ex-wife Demi Moore . However , he was also recently in the news for an altogether more disappointing reason , as he and Mila Kunis wrote a letter of support for their former co-star Danny Masterson while the latter was in court facing charges of rape . Masterson was convicted , and Kutcher and Kunis apologized , with Kutcher even -RRB- . It has been a rough few months for Trump , who seems to be losing even more than usual . Fresh from yet another E Jean Carroll for defamation -LRB- she successfully sued Trump in civil court for rape , and then took him to the cleaners again as he could n't stop publicly attacking her character -RRB- , his most current case has seen him dragged over the coals in a thoroughly embarrassing manner . This particular bit of legal trouble relates to an affair that Trump had with the adult film actress Stormy Daniels . The extramarital dalliance was alleged to have taken place while Trump 's present wife , Melania , was pregnant with their son , Barron . While that is all morally repugnant and grossly unethical , it is n't actually illegal . Where the former president and failed insurrectionist messed up was via the funds he used to pay for Daniels ' silence to increase his chances of winning the 2016 election , which he raised by falsifying business records . The payments in question were made through Trump 's former lawyer , Michael Cohen , who has admitted to this crime and to several others , and has been testifying in the case against his old boss . Cohen has already been convicted of a number of crimes that relate to his time as a Trump employee . One of the most ironic things about all of this is that a large chunk of Republican voters are so deep into the Trump cult that his affair being public knowledge would have likely had little bearing on the election . After all , Trump won vote in the 2020 election , despite all of this being widely known . And who can forget , his supporters are now as the advanced state of Trump 's incontinence issues are becoming public . Sadly , it seems Kutcher is not going to pop out from behind the scene and tell us we 're all being punk 'd , but there is no doubt that Trump continues to be a joke -- albeit a dangerous one .'''

    # coll = pymongo.MongoClient("172.31.29.170", 27017, unicode_decode_error_handler="ignore")["staticFeature"][
    #     "document"]
    # cursor = coll.find_one({"_id": "0s2fQQqt"})
    # for document in cursor:
    #     title = document["stitle"]
    #     content = document["seg_content"]
    #     break

    # prompt = _writing_quality_format_prompt(prompt=writing_quality_prompt, title=title, content=content)
    prompt = _writing_quality_format_prompt(prompt=finetune_prompt.writing_quality_without_rationale_prompt, title=title, content=content)
    # prompt = _writing_quality_format_prompt(prompt=finetune_prompt.content_genre_without_reason_template, title=title, content=content)
    print(prompt)

    # infer_obj = LLamaInfer()
    infer_obj = TGIInfer(port_in_use=8311)
    # infer_obj = TensorRTLLMInfer()

    start = datetime.datetime.now()

    answer = infer_obj.infer(prompt)
    # answer = infer_obj.infer_use_http(prompt)
    print(answer)
    print(json.dumps(answer, indent=2))

    # answer = infer_obj.client.text_generation(prompt=prompt, max_new_tokens=1024, details=True, top_k=1, temperature=0.01)
    # answer = infer_obj.client.text_generation(prompt=prompt, max_new_tokens=1024, details=True, best_of=2, do_sample=True)
    # print(answer.generated_text)
    # for one_token in answer.details.tokens:
    #     print(f"{one_token.text}\t{one_token.logprob}")

    end = datetime.datetime.now()
    time_deta = end - start
    time_ms = time_deta.seconds * 1000 + time_deta.microseconds / 1000
    print(f"cost time: {time_ms} ms")

    # print(f"######")
    # print(answer[0]['generated_text'])


def get_test_result_content_quality_by_file():
    # warning: should use correct test file
    input_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/rewrite_result3.jsonl"
    lines = [one_line.strip() for one_line in open(input_file, 'r', encoding="utf-8").readlines()]
    infer_obj = TGIInfer()

    output_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/rewrite_result4.jsonl"
    f_out = open(output_file, 'w', encoding="utf-8")

    num = 0
    for one_line in lines:
        one_json = json.loads(one_line)
        # prompt = one_json["prompt"].strip()
        # prompt = " ".join(prompt.split()[:1400])
        title = one_json["title"]
        rewrite_result = one_json["rewrite_result"]
        # rewrite_title = rewrite_result["Title"].strip()
        rewrite_content = rewrite_result["Content"].strip()
        rewrite_content = " ".join(rewrite_content.split()[:1400])
        prompt = finetune_prompt.writing_quality_prompt.format(title=title, content=rewrite_content)

        try:
            llama_result = infer_obj.infer(prompt)
        except Exception as e:
            print(f"Exception occurred: {e}, continue")
            continue
        one_json["rewrite_prompt"] = prompt
        one_json["llama_rewrite_result"] = llama_result
        one_json_str = json.dumps(one_json)

        f_out.write(one_json_str)
        f_out.write("\n")
        f_out.flush()
        num += 1
        print(num)

    f_out.close()


def get_test_result_unified_by_file():
    # warning: should use correct test file
    input_file = "/mnt/share16t/liujie/code_workspace/dataset/unified_dataset/unified_valid_0303.jsonl"
    lines = [one_line.strip() for one_line in open(input_file, 'r', encoding="utf-8").readlines()]
    infer_obj = TGIInfer()

    output_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/llama_result_unified.jsonl"
    f_out = open(output_file, 'w', encoding="utf-8")

    tokenizer = LlamaTokenizer.from_pretrained("/mnt/share16t/liujie/code_workspace/llama-recpies/exp/llama_13bf_multitask_v1")
    tokenizer.model_max_length = 4096
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = 0

    num = 0
    for one_line in lines:
        one_json = json.loads(one_line)
        inputs, labels = unified_dataset.process_dialog_to_single_turn(one_json, tokenizer)
        inputs_str = tokenizer.decode(inputs)

        b_index = inputs_str.find(B_INST)
        e_index = inputs_str.find(E_INST)
        prompt = inputs_str[b_index + len(B_INST): e_index].strip()
        gpt_result = inputs_str[e_index + len(E_INST): len(E_SENTENCE) * -1].strip()
        try:
            llama_result = infer_obj.infer(prompt)
        except Exception as e:
            print(f"Exception occurred: {e}, continue")
            continue
        one_json["llama_result"] = llama_result
        one_json["prompt_decoded"] = prompt
        one_json["gpt_result"] = gpt_result
        one_json_str = json.dumps(one_json)

        f_out.write(one_json_str)
        f_out.write("\n")
        f_out.flush()
        num += 1
        print(num)

    f_out.close()


def get_test_result_content_quality_format_by_prompt():
    input_file = "/mnt/share16t/liujie/code_workspace/dataset/creator_sample/local_nonlocal_news_sample.jsonl"
    lines = [one_line.strip() for one_line in open(input_file, 'r', encoding="utf-8").readlines()]
    infer_obj = TGIInfer()

    output_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/llama_result_local_nonlocal.jsonl"
    f_out = open(output_file, 'w', encoding="utf-8")

    num = 0
    for one_line in lines:
        one_json = json.loads(one_line)
        title = one_json["title"].strip()
        content = one_json["content"].strip()
        prompt = finetune_prompt.writing_quality_prompt.format(title=title, content=content)
        try:
            llama_result = infer_obj.infer(prompt)
        except Exception as e:
            print(f"Exception occurred: {e}, continue")
            continue
        one_json["prompt"] = prompt
        one_json["llama_result"] = llama_result
        one_json_str = json.dumps(one_json)

        f_out.write(one_json_str)
        f_out.write("\n")
        f_out.flush()
        num += 1
        print(num)

    f_out.close()


def parse_file():
    input_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/llama_result.jsonl"
    output_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/llama_result.tsv"
    f_output = open(output_file, 'w', encoding="utf-8")
    with open(input_file, 'r', encoding="utf-8") as f:
        for one_line in f:
            one_json = json.loads(one_line)
            doc_id = one_json["doc_id"]
            gpt_result = json.dumps(one_json["gpt_result"])
            llama_result = one_json["llama_result"]
            new_line = f"{doc_id}\t{gpt_result}\t{llama_result}"

            f_output.write(new_line)
            f_output.write("\n")
            f_output.flush()
    f_output.close()


def format_rouge_score(score):
    for k in ["rouge-1", "rouge-2", "rouge-l"]:
        if k in score:
            for sub_k in ["r", "p", "f"]:
                if sub_k in score[k]:
                    score[k][sub_k] = round(score[k][sub_k], 2)
    return score


def cal_rouge_score():
    is_print = True
    input_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/llama_result.jsonl"
    rationales = ["Clarity_and_Coherence_Evaluation_Rationale",
                  "Relevance_and_Focus_Evaluation_Rationale",
                  "Accuracy_and_Credibility_Evaluation_Rationale",
                  "Originality_and_Insightfulness_Evaluation_Rationale",
                  "Engagement_and_Interest_Evaluation_Rationale",
                  "Grammar_and_Style_Evaluation_Rationale",
                  "Structural_Integrity_Evaluation_Rationale",
                  "Argumentation_and_Evidence_Evaluation_Rationale",
                  "Overall_Evaluation_Rationale"]
    gpt_rational_dict = {}
    llama_rational_dict = {}
    for rationale in rationales:
        gpt_rational_dict[rationale] = []
        llama_rational_dict[rationale] = []

    with open(input_file, 'r', encoding="utf-8") as f:
        for one_line in f:
            one_json = json.loads(one_line.strip())
            gpt_result = one_json["gpt_result"]
            try:
                llama_result_str = one_json["llama_result"]
                llama_result = json.loads(llama_result_str)
            except json.decoder.JSONDecodeError:
                logger.warning(f"{llama_result_str} could not parse, continue")
                continue

            for rationale in rationales:
                if rationale not in gpt_result or rationale not in llama_result:
                    logger.warning(f"{rationale} not found in gpt or llama result, continue")
                    continue
                gpt_rational_dict[rationale].append(gpt_result[rationale].strip())
                llama_rational_dict[rationale].append(llama_result[rationale].strip())

    rouge_obj = Rouge()
    gpt_rational_all = []
    llama_rational_all = []
    for rationale in rationales:
        scores = rouge_obj.get_scores(hyps=llama_rational_dict[rationale], refs=gpt_rational_dict[rationale], avg=True)
        scores = format_rouge_score(scores)
        if is_print:
            print(f"{rationale}\t{json.dumps(scores)}")
        else:
            logger.info(f"rational: {rationale} , scores: {scores}")

        gpt_rational_all.extend(gpt_rational_dict[rationale])
        llama_rational_all.extend(llama_rational_dict[rationale])

    all_scores = rouge_obj.get_scores(hyps=llama_rational_all, refs=gpt_rational_all, avg=True)
    all_scores = format_rouge_score(all_scores)
    if is_print:
        print(f"all_score\t{json.dumps(all_scores)}")
    else:
        logger.info(f"all_scores: {all_scores}")


def cal_unified_rouge_score():
    is_print = True
    input_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/llama_result_unified.jsonl"

    gpt_results = []
    llama_results = []
    with open(input_file, 'r', encoding="utf-8") as f:
        for one_line in f:
            one_json = json.loads(one_line.strip())
            gpt_result = one_json["gpt_result"]
            if len(gpt_result) == 0:
                logger.info(f"gpt_result is empty, continue")
                continue
            llama_result = one_json["llama_result"]
            if len(llama_result) == 0:
                logger.info(f"llama_result is empty, continue")
                continue

            gpt_results.append(gpt_result)
            llama_results.append(llama_result)

    logger.info(f"gpt_results: {len(gpt_results)}")
    logger.info(f"llama_results: {len(llama_results)}")
    rouge_obj = Rouge()
    all_scores = rouge_obj.get_scores(hyps=llama_results, refs=gpt_results, avg=True)
    all_scores = format_rouge_score(all_scores)
    if is_print:
        print(f"all_score\t{json.dumps(all_scores)}")
    else:
        logger.info(f"all_scores: {all_scores}")


def cal_consistent_score():
    is_print = True
    input_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/llama_result.jsonl"
    scores = ["Clarity_and_Coherence_Score",
              "Relevance_and_Focus_Score",
              "Accuracy_and_Credibility_Score",
              "Originality_and_Insightfulness_Score",
              "Engagement_and_Interest_Score",
              "Grammar_and_Style_Score",
              "Structural_Integrity_Score",
              "Argumentation_and_Evidence_Score",
              "Overall_Score"]
    gpt_score_dict = {}
    llama_score_dict = {}
    for score in scores:
        gpt_score_dict[score] = []
        llama_score_dict[score] = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for one_line in f:
            one_json = json.loads(one_line.strip())
            gpt_result = one_json["gpt_result"]
            try:
                llama_result_str = one_json["llama_result"]
                llama_result_origin = json.loads(llama_result_str)
            except json.decoder.JSONDecodeError:
                logger.warning(f"{llama_result_str} could not parse, continue")
                continue

            # clean data, for robust
            llama_result = {}
            for k, v in llama_result_origin.items():
                llama_result[k.strip()] = v

            for score in scores:
                if score not in gpt_result or score not in llama_result:
                    logger.warning(f"{score} not found in gpt or llama result, continue")
                    continue
                gpt_score_dict[score].append(gpt_result[score])
                llama_score_dict[score].append(llama_result[score])

    gpt_score_all = []
    llama_score_all = []
    for score in scores:
        gpt_score = gpt_score_dict[score]
        llama_score = llama_score_dict[score]
        consistent = 0
        for left, right in zip(gpt_score, llama_score):
            if left == right:
                consistent += 1
        ratio = consistent / len(gpt_score)
        ratio = round(ratio, 2)
        if is_print:
            print(f"{score}\t{ratio}")
        else:
            logger.info(f"{score} consistent: {consistent}, ratio: {ratio}")

        gpt_score_all.extend(gpt_score)
        llama_score_all.extend(llama_score)

    consistent_all = 0
    for left, right in zip(gpt_score_all, llama_score_all):
        if left == right:
            consistent_all += 1
    ratio_all = consistent_all / len(gpt_score_all)
    ratio_all = round(ratio_all, 2)
    if is_print:
        print(f"ratio_all\t{ratio_all}")
    else:
        logger.info(f"all consistent: {consistent_all}, ratio: {ratio_all}")


def flat_train_file():
    input_file = "/mnt/share16t/liujie/code_workspace/dataset/content_quality_dataset/train.jsonl"
    output_file = "/mnt/share16t/liujie/code_workspace/dataset/content_quality_dataset/flat_train.jsonl"

    rationales = ["Clarity_and_Coherence_Evaluation_Rationale",
                  "Relevance_and_Focus_Evaluation_Rationale",
                  "Accuracy_and_Credibility_Evaluation_Rationale",
                  "Originality_and_Insightfulness_Evaluation_Rationale",
                  "Engagement_and_Interest_Evaluation_Rationale",
                  "Grammar_and_Style_Evaluation_Rationale",
                  "Structural_Integrity_Evaluation_Rationale",
                  "Argumentation_and_Evidence_Evaluation_Rationale"]
    scores = ["Clarity_and_Coherence_Score",
              "Relevance_and_Focus_Score",
              "Accuracy_and_Credibility_Score",
              "Originality_and_Insightfulness_Score",
              "Engagement_and_Interest_Score",
              "Grammar_and_Style_Score",
              "Structural_Integrity_Score",
              "Argumentation_and_Evidence_Score"]

    rational_explain_map = {
        "Clarity_and_Coherence_Evaluation_Rationale": "Clarity and Coherence: The ability of the article to communicate ideas in a clear and logical manner. This includes the use of straightforward language, well-structured sentences, and a coherent flow of ideas.",
        "Relevance_and_Focus_Evaluation_Rationale": "Relevance and Focus: The article should stay focused on the central topic or argument without diverging into unrelated areas. Relevance ensures that all parts of the article contribute directly to its main purpose or thesis.",
        "Accuracy_and_Credibility_Evaluation_Rationale": "Accuracy and Credibility: Factual accuracy is crucial for building trust with readers. This involves verifying facts, citing reliable sources, and providing accurate representations of data and events.",
        "Originality_and_Insightfulness_Evaluation_Rationale": "Originality and Insightfulness: Original content that provides new insights, perspectives, or findings adds significant value. This dimension assesses the uniqueness of the article and its contribution to the subject matter.",
        "Engagement_and_Interest_Evaluation_Rationale": "Engagement and Interest: The ability to capture and hold the reader's attention through compelling writing, interesting anecdotes, or thought-provoking questions. Engaging writing often includes a strong introduction, dynamic content, and a memorable conclusion.",
        "Grammar_and_Style_Evaluation_Rationale": "Grammar and Style: Proper grammar, punctuation, and spelling are fundamental to writing quality. Style also plays a role, including the choice of words, tone, and the use of literary devices that enhance the reading experience.",
        "Structural_Integrity_Evaluation_Rationale": "Structural Integrity: The organization of the article, including the use of headings, subheadings, paragraphs, and bullet points, which can help in making the content more digestible and easier to follow.",
        "Argumentation_and_Evidence_Evaluation_Rationale": "Argumentation and Evidence: For articles that aim to persuade or present an argument, the quality of reasoning, the strength of evidence provided, and the effectiveness of the argumentative structure are important. ",
    }

    prompt_template = '''We measure an article's writing quality based on the following
Given a article title and content below, please evaluate its writing quality based on the following dimension:
{rational_explain}
Below is an article title and content:
Title: {title}
Content: {content}
Please evaluate its write quality, assign a score (1 to 5) for this dimension.
Please output with the following json format 
{{"{score_key}": score (1 to 5), "{rationale_key}": XXX}}
Please output now:
'''

    f_out = open(output_file, "w", encoding="utf-8")
    with open(input_file, "r", encoding="utf-8") as f:
        for one_line in f:
            one_json = json.loads(one_line.strip())
            gpt_result = one_json["gpt_result"]
            for rationale_key, score_key in zip(rationales, scores):
                if rationale_key not in gpt_result or score_key not in gpt_result:
                    logger.warning(f"{rationale_key} or {score_key} not in gpt_result, continue")
                    continue
                rationale_value = gpt_result[rationale_key]
                score_value = gpt_result[score_key]
                title = one_json["title"]
                content = one_json["content"]

                rational_explain = rational_explain_map[rationale_key]
                prompt_new = prompt_template.format(rational_explain=rational_explain, title=title, content=content,
                                                    score_key=score_key, rationale_key=rationale_key)
                gpt_result_new = {
                    score_key: score_value,
                    rationale_key: rationale_value
                }

                new_json = copy.deepcopy(one_json)
                new_json["prompt"] = prompt_new
                new_json["gpt_result"] = gpt_result_new
                new_json_str = json.dumps(new_json)

                f_out.write(new_json_str)
                f_out.write("\n")
                f_out.flush()

    f_out.close()


def parse_tsv_file():
    input_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/detrimental_sample_result.jsonl"
    output_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/detrimental_sample_result.tsv"

    # input_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/creator_sample_result.jsonl"
    # output_file = "/mnt/share16t/liujie/code_workspace/llama-recpies/creator_sample_result.tsv"

    f_out = open(output_file, "w", encoding="utf-8")

    score_keys = ["Clarity_and_Coherence_Score",
              "Relevance_and_Focus_Score",
              "Accuracy_and_Credibility_Score",
              "Originality_and_Insightfulness_Score",
              "Engagement_and_Interest_Score",
              "Grammar_and_Style_Score",
              "Structural_Integrity_Score",
              "Argumentation_and_Evidence_Score",
              "Overall_Score"]

    score_str = "\t".join(score_keys)
    one_line = f"first_column\tdoc_id\t{score_str}"
    f_out.write(one_line + "\n")
    f_out.flush()

    with open(input_file, "r", encoding="utf-8") as f_in:
        for one_line in f_in:
            one_line = one_line.strip()
            one_json = json.loads(one_line)

            doc_id = one_json["doc_id"]
            # first_column = one_json["media_id"]
            first_column = one_json["detrimental_score"]

            # llama_result_json = one_json["gpt_result"]
            llama_result_json = json.loads(one_json["llama_result"])

            # title = one_json["title"]
            # content = one_json["content"]
            # llama_result = one_json["llama_result"]

            scores = []
            for key in score_keys:
                score = llama_result_json[key]
                scores.append(str(score))
            score_str = "\t".join(scores)

            one_line = f"{first_column}\t{doc_id}\t{score_str}"
            f_out.write(one_line + "\n")
            f_out.flush()
    f_out.close()


def insert_task_type_by_file(input_file, task_type):
    results = []
    with open(input_file, "r", encoding="utf-8") as f_in:
        for one_line in f_in:
            one_line = one_line.strip()
            one_json = json.loads(one_line.strip())
            one_json["task_type"] = task_type
            results.append(one_json)
    return results

def write_json_to_file(output_file, json_results):
    with open(output_file, "w", encoding="utf-8") as f_out:
        for one_json in json_results:
            f_out.write(json.dumps(one_json) + "\n")

def get_multitask_train_data():
    # content_quality_input_file = "/mnt/share16t/liujie/code_workspace/dataset/content_quality_dataset/train_0607.jsonl"
    # unified_input_file = "/mnt/share16t/liujie/code_workspace/dataset/unified_dataset/unified_train_0303.jsonl"
    # content_genre_input_file = "/mnt/share16t/liujie/code_workspace/dataset/content_genre_dataset/genre_train3.jsonl"
    # multitask_output_file = "/mnt/share16t/liujie/code_workspace/dataset/multitask_dataset/multitask_train_0607.jsonl"

    content_quality_input_file = "/mnt/share16t/liujie/code_workspace/dataset/content_quality_dataset/test.jsonl"
    unified_input_file = "/mnt/share16t/liujie/code_workspace/dataset/unified_dataset/unified_valid_0303.jsonl"
    content_genre_input_file = "/mnt/share16t/liujie/code_workspace/dataset/content_genre_dataset/genre_test3.jsonl"
    multitask_output_file = "/mnt/share16t/liujie/code_workspace/dataset/multitask_dataset/multitask_test_0607.jsonl"

    content_results = insert_task_type_by_file(content_quality_input_file, "content_quality")
    second_results = insert_task_type_by_file(content_genre_input_file, "content_genre")

    all_results = []
    all_results.extend(content_results)
    all_results.extend(second_results)

    random.shuffle(all_results)

    write_json_to_file(multitask_output_file, all_results)

def calc_pr(label2count, pred2count, tp2count):
    print("label:", label2count)
    print("pred:", pred2count)
    print("true positive:", tp2count)
    print("label\tprec\trecall\tf1")
    for label, count in label2count.items():
        pred_count = pred2count.get(label, 0)
        tp_count = tp2count.get(label, 0)
        if pred_count != 0:
            prec = tp_count*1.0/pred_count
        else:
            prec = 0
        recall = tp_count*1.0/count
        if prec + recall == 0:
            f1 = 0
        else:
            f1 = 2* prec * recall / (prec + recall)
        print(f"{label}\t{prec}\t{recall}\t{f1}")
    print("\n###########\n")


def infer_content_genre_metric():
    fn = "/mnt/share16t/liujie/code_workspace/dataset/content_genre_dataset/genre_test3.jsonl"
    tgi = TGIInfer()
    ret = []
    import collections
    label2count = collections.defaultdict(int)
    pred2count = collections.defaultdict(int)
    tp2count = collections.defaultdict(int)
    i = 0
    for line in open(fn):
        data = json.loads(line)
        title = data["title"]
        title = prompt_util.get_by_limit_len_title(title)
        content = data["content"]
        content = prompt_util.get_by_limit_len_content(content)
        prompt = finetune_prompt.content_genre_without_reason_template.format(title=title, content=content)
        try:
            reply = tgi.infer(prompt)
            reply_json = json.loads(reply)
        except Exception as e:
            logger.error(e)
            logger.info(prompt)
            continue
        label = data["gpt_result"]["category"]
        pred = reply_json["category"]
        label2count[label] += 1
        pred2count[pred] += 1
        if label.lower() == pred.lower():
            tp2count[label] += 1
        i += 1
        if i % 50 == 0:
            calc_pr(label2count, pred2count, tp2count)
        data["pred"] = reply.strip()
        ret.append(json.dumps(data) + "\n")
    print("final:")
    calc_pr(label2count, pred2count, tp2count)
    output_file = fn + ".pred"
    open(output_file, "w").writelines(ret)
    content_genre_metric(output_file)

def content_genre_metric(input_file):
    label_to_score = {
        "News": 0,
        "Investigative": 1,
        "Opinion": 2,
        "Interview": 3,
        "Profile": 4,
        "Review": 5,
        "Guide": 6,
        "Analysis": 7,
        "Essay": 8,
        "Satire": 9,
    }

    label_list = []
    predict_list = []
    json_lines = [json.loads(line.strip()) for line in open(input_file, "r", encoding="utf-8").readlines()]
    for one_json in json_lines:
        if "pred" in one_json and type(one_json["pred"]) == str:
            one_json["pred"] = json.loads(one_json["pred"])
        if ("gpt_result" in one_json and "category" in one_json["gpt_result"] and one_json["gpt_result"]["category"] in label_to_score
                and "pred" in one_json and "category" in one_json["pred"] and one_json["pred"]["category"] in label_to_score):
            label_genre = one_json["gpt_result"]["category"]
            label = label_to_score[label_genre]
            label_list.append(label)

            predict_genre = one_json["pred"]["category"]
            predict = label_to_score[predict_genre]
            predict_list.append(predict)
        else:
            logger.warning(f"one_json is not valid, continue")
    # warning, tmp fix
    # del label_to_score["Investigative"]
    names = list(label_to_score.keys())

    report_str = metrics.classification_report(y_true=label_list, y_pred=predict_list, target_names=names)
    print(report_str)


def infer_speed_multi_process(model, parallel_n):
    infer_speed_obj = InferSpeed()
    infer_speed_obj.run_infer_speed(model, parallel_n)


def infer_llama_multi_process(model, parallel_n, input_file, output_file):
    infer_llama_obj = MultiProcessInfer()
    infer_llama_obj.run_infer_multi_process(model, parallel_n, input_file, output_file)


def test_tokenizer():
    llama2_path = "meta-llama/Llama-2-13b-hf"
    llama3_path = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(llama3_path)
    input_ids = tokenizer.encode(finetune_prompt.content_genre_template)
    print(type(input_ids))
    print(len(input_ids))
    print(input_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str)
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--parallel_n', type=int, required=False)
    parser.add_argument('--input_file', type=str, required=False)
    parser.add_argument('--output_file', type=str, required=False)
    args = parser.parse_args()
    # python3 llama_infer.py --job infer_speed_multi_process --model tgi --parallel_n 20
    if args.job == "infer_speed_multi_process":
        infer_speed_multi_process(model=args.model, parallel_n=args.parallel_n)
    # python3 llama_infer.py --job infer_llama_multi_process --model tgi --parallel_n 64 \
    #   --input_file "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240401.low_quality.jsonl" \
    #   --output_file "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/train.20240401.low_quality.writing_quality.jsonl"
    elif args.job == "infer_llama_multi_process":
        infer_llama_multi_process(model=args.model, parallel_n=args.parallel_n, input_file=args.input_file, output_file=args.output_file)
    # python3 llama_infer.py --job parse_to_tsv
    elif args.job == "parse_to_tsv":
        parse_tsv_file()
    # python3 llama_infer.py --job get_multitask_train_data
    elif args.job == "get_multitask_train_data":
        get_multitask_train_data()
    # python3 llama_infer.py --job infer_content_genre_metric
    elif args.job == "infer_content_genre_metric":
        infer_content_genre_metric()
    # python3 llama_infer.py --job content_genre_metric
    elif args.job == "content_genre_metric":
        content_genre_metric(input_file="/mnt/share16t/liujie/code_workspace/dataset/content_genre_dataset/genre_test3.jsonl.pred")
    # python3 llama_infer.py --job test_one
    elif args.job == "test_one":
        test_one()
    # python3 llama_infer.py --job tmp_job
    elif args.job == "tmp_job":
        pass
        # test_tokenizer()
        # get_test_result_content_quality_by_file()
        # cal_rouge_score()
        # cal_consistent_score()
        # get_test_result_unified_by_file()
        # cal_unified_rouge_score()
        # parse_file()
        # flat_train_file()
        # parse_tmp_file()
        # cal_unified_rouge_score()
        # infer_speed_multi_process()
        # get_test_result_content_quality_format_by_prompt()
    else:
        raise NotImplementedError(f"{args.job} is not implemented")
