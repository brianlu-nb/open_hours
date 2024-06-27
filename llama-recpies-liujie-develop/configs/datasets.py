# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, asdict

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"


@dataclass
class yelp_dataset:
    dataset: str = 'yelp_dataset'
    train_split: str = "ft_datasets/new_yelp_dialog_train.jsonl"
    test_split: str = "ft_datasets/new_yelp_dialog_valid.jsonl"
    system_template = (
        "Below is some background information: "
        "{system}"
        "Answer questinos based on the given info, and provide the reference No. in the end of you answer."
        "If the question cannot be answered with the information given, please output a search query beginning with 'query:'"
    )
    use_system_prompt: bool = False
    only_first_turn: bool = False


@dataclass
class yp_dataset:
    dataset: str = 'yp_dataset'
    train_split: str = "ft_datasets/yp_train.jsonl"
    test_split: str = "ft_datasets/yp_valid.jsonl"
    system_template = None
    use_system_prompt: bool = False
    only_first_turn: bool = False

@dataclass
class unified_dataset:
    dataset: str = 'unified_dataset'
    train_split: str = "data/unified_train_0303.jsonl"
    test_split: str = "data/unified_valid_0303.jsonl"
    train_on_context: float = -1
    shuffle_ref: bool = False

@dataclass
class unified_dpo_dataset:
    dataset: str = 'unified_dpo_dataset'
    # train_split: str = "data/unified_train_dpo_1016.jsonl"
    # test_split: str = "data/unified_valid_dpo_1016.jsonl"
    # train_split: str = "data/gy_dpo_1210_better_train.jsonl"
    # test_split: str = "data/gy_dpo_1210_better_valid.jsonl"
    train_split: str = "data/gy_dpo_1211_winnerall_train.jsonl"
    test_split: str = "data/gy_dpo_1211_winnerall_valid.jsonl"

@dataclass
class article_id_dataset:
    dataset: str = 'article_id_dataset'
    train_split: str = "/home/paperspace/chengniu/llama-recpies/datasets/train.txt"
    test_split: str = "/home/paperspace/chengniu/llama-recpies/datasets/valid.txt"

@dataclass
class youtube_teaser_dataset:
    dataset: str = 'youtube_teaser_dataset'
    train_split: str = "/home/paperspace/chengniu/llama-recpies/datasets/youtube_teaser_0.train.json"
    test_split: str = "/home/paperspace/chengniu/llama-recpies/datasets/youtube_teaser_0.test.json"


@dataclass
class content_quality_dataset:
    dataset: str = 'content_quality_dataset'
    train_split: str = "/mnt/share16t/liujie/code_workspace/dataset/content_quality_dataset/train.jsonl"
    test_split: str = "/mnt/share16t/liujie/code_workspace/dataset/content_quality_dataset/test.jsonl"


if __name__=='__main__':
    from transformers import TrainingArguments
    from typing import Dict, Optional, Sequence, List
    from dataclasses import dataclass, field, asdict
    import transformers

    @dataclass
    class TrainingArguments(transformers.TrainingArguments):
        dataset_config: dict = field(default_factory=dict)
        cache_dir: Optional[str] = field(default=None)
        optim: str = field(default="adamw_torch")
        model_max_length: int = field(
            default=4096,
            metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
        )
        use_peft: Optional[bool] = field(default=False)
        peft_method: Optional[str] = field(default='lora')
        lora_r: Optional[int] = field(default=8)
        lora_alpha: Optional[int]= field(default=32)
        target_modules: str = field(default="q_proj,v_proj")
        bias: Optional[str] = field(default="none")
        task_type: Optional[str] = field(default="CAUSAL_LM")
        lora_dropout: Optional[float]= field(default=0.05)
        inference_mode: Optional[bool] = field(default=False)
    # args = TrainingArguments('test')
    c = unified_dataset()
    # print(asdict(c))
    # args.dataset_config = asdict(c)
    # print(args)
    print(str(c))
