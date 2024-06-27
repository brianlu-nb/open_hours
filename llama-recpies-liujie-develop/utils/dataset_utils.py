# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
import torch

from functools import partial

from ft_datasets import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
    get_yelp_dataset,
    get_unified_dataset,
    get_unified_dpo_dataset,
    get_article_id_dataset,
    get_youtube_teaser_dataset,
    get_content_quality_dataset
)
from typing import Optional


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=224),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "yelp_dataset": get_yelp_dataset,
    "yp_dataset": get_yelp_dataset,
    "unified_dataset": get_unified_dataset,
    "unified_dpo_dataset": get_unified_dpo_dataset,
    "article_id_dataset": get_article_id_dataset,
    "youtube_teaser_dataset": get_youtube_teaser_dataset,
    "content_quality_dataset": get_content_quality_dataset
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )


@dataclass
class PaddingCollactor:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    input_pad_id: int
    label_pad_id: int = -100
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = 0
        for f in features:
            max_len = max(max_len, len(f['input_ids']))
        input_ids = torch.ones((len(features), max_len), dtype=torch.long)*self.input_pad_id
        labels = torch.ones((len(features), max_len), dtype=torch.long)*self.label_pad_id
        for idx, f in enumerate(features):
            input_ids[idx, :len(f['input_ids'])] = torch.tensor(f['input_ids'], dtype=torch.long)
            labels[idx, :len(f['labels'])] = torch.tensor(f['labels'], dtype=torch.long)

        if self.max_length is not None and self.max_length<max_len:
            input_ids = input_ids[:,:self.max_length]
            labels = labels[:, :self.max_length]
        batch = {
            'input_ids': input_ids,
            'labels': labels
        }
        return batch