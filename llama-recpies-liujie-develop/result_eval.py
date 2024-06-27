import json
import random
from json import JSONDecodeError
from utils.mongo_util import MongoUtil
import tqdm
import logging


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ResultEval(object):

    def __init__(self):
        self.scores = [
            "Clarity_and_Coherence_Score",
            "Relevance_and_Focus_Score",
            "Accuracy_and_Credibility_Score",
            "Originality_and_Insightfulness_Score",
            "Engagement_and_Interest_Score",
            "Grammar_and_Style_Score",
            "Structural_Integrity_Score",
            "Argumentation_and_Evidence_Score",
            "Overall_Score"]
        self.dim_scores = [
            "Clarity_and_Coherence_Score",
            "Relevance_and_Focus_Score",
            "Accuracy_and_Credibility_Score",
            "Originality_and_Insightfulness_Score",
            "Engagement_and_Interest_Score",
            "Grammar_and_Style_Score",
            "Structural_Integrity_Score",
            "Argumentation_and_Evidence_Score"]

    def read_json(self, input_file: str):
        json_list = []
        with open(input_file, "r", encoding="utf-8") as f:
            for one_line in f.readlines():
                one_line = one_line.strip()
                json_list.append(json.loads(one_line))
        return json_list

    def valid(self, one_json: dict):
        if "llama_result" not in one_json:
            return False
        llama_result = one_json["llama_result"]
        if type(llama_result) == str:
            llama_result = json.loads(llama_result)
        for key in self.dim_scores:
            if key not in llama_result:
                return False
            score = llama_result[key]
            if type(score) not in [int, float]:
                return False
        return True

    def parse(self, one_json: dict):
        llama_result = one_json["llama_result"]
        if type(llama_result) == str:
            llama_result = json.loads(llama_result)
            one_json["llama_result"] = llama_result
        score_sum = 0.0
        for key in self.dim_scores:
            score = llama_result[key]
            score_sum += score
        score_avg = score_sum / len(self.dim_scores)
        one_json["llama_result"]["score_avg"] = score_avg
        return one_json

    def sort_by_score(self, json_input: list):
        json_list = []
        for one_json in json_input:
            if self.valid(one_json):
                one_json = self.parse(one_json)
                json_list.append(one_json)
        logger.info(f"json_list len: {len(json_list)}")

        json_list.sort(key=lambda x: x["llama_result"]["score_avg"])
        return json_list


    def sample_resluts(self, json_list: list):
        min_score = 1.0
        max_score = 4.7
        step = (max_score - min_score) / 100.0
        score_dict = {}
        for i in range(100):
            score_dict[i] = []
        for one_json in json_list:
            content = one_json.get("seg_content", "")
            content_split = content.split(" ")
            if len(content_split) < 50:
                continue
            avg_score = one_json["llama_result"]["score_avg"]
            index = int((avg_score - min_score) / step)
            score_dict[index].append(one_json)

        json_sampled = []
        for _, v in score_dict.items():
            sample_len = min(len(v), 4)
            if sample_len == 0:
                continue
            json_sampled.extend(random.sample(v, sample_len))
        logger.info(f"json_sampled len: {len(json_sampled)}")
        return json_sampled

    def enrich_static_feature(self, json_list: list):
        static_feature_coll = MongoUtil.get_coll_static_feature()
        project_field_dict = {k: True for k in ["content_type", "local_score", "richness_score"]}
        pbar = tqdm.tqdm(total=len(json_list))
        num = 0
        for one_json in json_list:
            num += 1
            if num % 100 == 0:
                pbar.update(100)
            doc_id = one_json["doc_id"]
            doc = static_feature_coll.find_one({"_id": doc_id}, project_field_dict)
            if doc is None:
                continue
            content_type = doc.get("content_type", "")
            local_score = doc.get("local_score", -1)
            richness_score = doc.get("richness_score", -1)
            one_json["content_type"] = content_type
            one_json["local_score"] = local_score
            one_json["richness_score"] = richness_score

    def filter(self, json_list: list):
        res_list = []
        for one_json in json_list:
            if one_json.get("content_type", "") != "news":
                continue
            if one_json.get("local_score", -1) < 0.5:
                continue
            res_list.append(one_json)
        return res_list

    def dedup_by_file(self, json_list: list, dedup_file: str):
        doc_id_set = set([one_line.strip() for one_line in open(dedup_file, 'r', encoding='utf-8').readlines()])
        res_json = []
        for one_json in json_list:
            doc_id = one_json["doc_id"]
            if doc_id in doc_id_set:
                continue
            res_json.append(one_json)
        return res_json

    def sample_results_by_score_bucket(self, json_list: list):
        json_list = filter(lambda x: x["llama_result"]["score_avg"] > 2.0 and x["llama_result"]["score_avg"] < 3.0, json_list)
        json_list = list(json_list)
        score_dict = {}
        for one_json in json_list:
            llama_result = one_json["llama_result"]
            num = self._cal_score_2_num(input_json=llama_result)
            num += self._richness_num(input_json=one_json)
            if num in score_dict:
                score_dict[num].append(one_json)
            else:
                score_dict[num] = []
                score_dict[num].append(one_json)
        score_list = sorted(score_dict.items(), key=lambda item: item[0], reverse=True)
        json_sampled = []
        for _, v in score_list:
            sample_len = min(len(v), 50)
            if sample_len == 0:
                continue
            json_sampled.extend(random.sample(v, sample_len))
        logger.info(f"json_sampled len: {len(json_sampled)}")
        return json_sampled


    def output_tsv(self, json_list: list, output_file: str):
        f_out = open(output_file, "w", encoding="utf-8")
        # names = ["score_2_num", "writing_quality_avg_score", "doc_id", "url", "seg_title", "seg_content", "reason",
        #          "status", "richness_score", "writing_quality_result", "diff", "gpt_result", "all"]
        # names = ["score_2_num", "writing_quality_avg_score", "doc_id", "url", "seg_title", "seg_content", "reason",
        #          "status", "richness_score", "writing_quality_result", "all"]
        names = ["writing_quality_avg_score", "doc_id", "media_id", "url", "title", "writing_quality_result"]
        f_out.write("\t".join(names) + "\n")
        for one_json in json_list:
            # score_2_num = self._cal_score_2_num(input_json=one_json["llama_result"])
            # score_2_num += self._richness_num(input_json=one_json)
            # score_2_num = str(score_2_num)
            avg_score = one_json["llama_result"]["score_avg"]
            doc_id = one_json["doc_id"]
            media_id = one_json["media_id"]
            url = one_json["url"]
            # url = "www.newsbreak.com/n/" + doc_id
            title = one_json["title"]
            # seg_title = one_json["seg_title"]
            # seg_content = one_json["seg_content"]
            # reason = one_json["reason"]
            # richness_score = str(one_json["richness_score"])
            # status = one_json["status"]
            llama_result = one_json["llama_result"]
            # diff = str(one_json["diff"])
            # gpt_result = one_json["gpt_result"]
            all = one_json
            # fields = [score_2_num, avg_score, doc_id, url, seg_title, seg_content, reason,
            #           status, richness_score, llama_result, diff, gpt_result, all]
            # fields = [score_2_num, avg_score, doc_id, url, seg_title, seg_content, reason,
            #           status, richness_score, llama_result, all]
            fields = [avg_score, doc_id, media_id, url, title, llama_result]
            fields = [json.dumps(f) if type(f) == dict else f for f in fields]
            fields = [str(f) if type(f) == float or type(f) == int else f for f in fields]
            f_out.write("\t".join(fields) + "\n")

    def output_jsonl(self, json_list: list, output_file: str):
        f_out = open(output_file, "w", encoding="utf-8")
        for one_json in json_list:
            f_out.write(json.dumps(one_json) + "\n")

    def _cal_score_2_num(self, input_json: dict):
        num = 0
        for key in self.dim_scores:
            dim_score = input_json[key]
            if dim_score == 2.0:
                num += 1
        return num

    def _richness_num(self, input_json: dict):
        num = 0
        richness = input_json.get("richness_score", -1)
        if richness == 0 or richness == 1:
            num = 1
        return num


    def parse_gpt_llama_diff(self, json_list: list):
        for one_json in json_list:
            gpt_result = one_json.get("gpt_result", {})
            delete_key = []
            for k, v in gpt_result.items():
                if k not in self.scores:
                    delete_key.append(k)
            for k in delete_key:
                del gpt_result[k]
            one_json["gpt_result"] = gpt_result

            llama_result = one_json.get("llama_result", {})
            diff = self._score_2_diff(llama_result, gpt_result)
            one_json["diff"] = diff

    def _score_2_diff(self, llama_result: dict, gpt_result: dict):
        llama_score_2 = self._cal_score_2_num(llama_result)
        gpt_score_2 = self._cal_score_2_num(gpt_result)
        return abs(llama_score_2 - gpt_score_2)

    def statistics_2_score_by_key(self, json_list: list):
        key_count = {}
        for key in self.dim_scores:
            key_count[key] = 0
        for one_json in json_list:
            for key in self.dim_scores:
                if key in one_json:
                    value = one_json[key]
                    if value == 2.0:
                        key_count[key] += 1

        key_list = sorted(key_count.items(), key=lambda k: k[1], reverse=True)
        for k, v in key_list:
            print(f"{k}\t{v}")

    def simple_convert(self, json_list: list):
        new_json_list = []
        for one_json in json_list:
            writing_quality_score = one_json["writing_quality_score"]
            one_json["llama_result"] = writing_quality_score
            new_json_list.append(one_json)
        return new_json_list


# python3 result_eval.py
if __name__ == '__main__':
    result_eval = ResultEval()
    # input_file = "/mnt/share16t/liujie/code_workspace/dataset/quality_dataset/quality.v34.visual/writing_quality.jsonl"
    input_file = "./creator_writing_quality_sample.jsonl"
    json_list = result_eval.read_json(input_file=input_file)
    # result_eval.statistics_2_score_by_key(json_list)

    # result_eval.parse_gpt_llama_diff(json_list=json_list)

    json_list = result_eval.simple_convert(json_list)
    json_sorted = result_eval.sort_by_score(json_list)

    # result_eval.enrich_static_feature(json_list=json_sorted)
    # logger.info(f"filter before: {len(json_sorted)}")
    # json_sorted = result_eval.filter(json_list=json_sorted)
    # logger.info(f"filter after: {len(json_sorted)}")
    # json_sorted = result_eval.dedup_by_file(
    #     json_list=json_sorted,
    #     dedup_file="/mnt/nlp/liujie/data/writing_quality_dedup.jsonl")
    # logger.info(f"filter after: {len(json_sorted)}")
    # json_sampled = result_eval.sample_results_by_score_bucket(json_sorted)
    output_file = "./creator_sampled.tsv"
    result_eval.output_tsv(json_list=json_sorted, output_file=output_file)
    # result_eval.output_jsonl(json_list=json_sampled, output_file=output_file)

    # for one_json in json_sorted:
    #     print(json.dumps(one_json, indent=2))
