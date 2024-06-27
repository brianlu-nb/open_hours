from transformers import AutoTokenizer
import json
import logging
import sys
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
print(FILE_PATH)
sys.path.append(FILE_PATH + "/../")

from ft_datasets import content_quality_dataset


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

LABEL_PAD_TOKEN_ID = -100


def test_one():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", trust_remote_code=True)
    lines = []
    input_file = "/mnt/share16t/liujie/code_workspace/dataset/content_quality_dataset/train.jsonl"
    with open(input_file, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    logger.info(f"lines len: {len(lines)}")
    data = lines[0]
    prompt = data["prompt"]
    gpt_result = json.dumps(data["gpt_result"])
    logger.info(f"prompt: {prompt}")
    logger.info(f"gpt_result: {gpt_result}")

    data["prompt"] = "Given a article title and content below"
    data["gpt_result"] = "{\"Clarity_and_Coherence_Score\": 4}"

    dataset_obj = content_quality_dataset.ContentQualityDataset(None, tokenizer, split_name=input_file)

    input_ids_api, label_ids_api = dataset_obj.process_using_api(data, tokenizer)

    input_ids, label_ids = dataset_obj.process(data, tokenizer)

    logger.info(f"input_ids: \n{input_ids}")
    logger.info(f"input_ids_api: \n{input_ids_api}")

    logger.info(f"input_words: \n{tokenizer.decode(input_ids)}")
    logger.info(f"input_api_words: \n{tokenizer.decode(input_ids_api)}")

    logger.info(f"#####")
    logger.info(f"label_ids: \n{label_ids}")
    logger.info(f"label_ids_api: \n{label_ids_api}")

    label_ids_api = list(filter(lambda x: x != -100, label_ids_api))

    logger.info(f"label_api_words: \n{tokenizer.decode(label_ids_api)}")


if __name__ == '__main__':
    test_one()