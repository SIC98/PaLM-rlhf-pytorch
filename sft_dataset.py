# Reference: `https://colab.research.google.com/drive/1_aws1VolXkvd4xIrFExTdc3qd1hm7nNv?usp=sharing#scrollTo=4E6u3j_YSMVX`
from typing import Optional, Dict, Sequence
from torch.utils.data import Dataset
import transformers
import logging
import json
import torch
import copy

from datasets import arrow_dataset

IGNORE_INDEX = -100


class SFT_dataset(Dataset):
    """SFT dataset by wygo"""

    def __init__(
        self,
        dataset: arrow_dataset.Dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        verbose=False,
        BOU_TOKEN="<|bou_token|>",
        BOC_TOKEN="<|boc_token|>",
    ):
        super(SFT_dataset, self).__init__()
        logging.warning("Loading data...")

        ## format
        pattern_instruction = "prompt"
        pattern_output = "messages"

        sources = []
        for example in dataset:
            sources.append(f"{example[pattern_instruction]}{BOU_TOKEN}")

        targets = []
        for example in dataset:
            targets.append(f"{example[pattern_output]}{BOC_TOKEN}")

        if verbose:
            idx = 0
            print((sources[idx]))
            print((targets[idx]))
            print("Tokenizing inputs... This may take some time...")

        examples = [s + t for s, t in zip(sources, targets)]

        # source data tokenized
        sources_tokenized = self._tokenize_fn(sources, tokenizer)
        examples_tokenized = self._tokenize_fn(examples, tokenizer)

        ## 입력은 source, 출력은 source+target 이지만 학습은 target 부분만
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX  # source 부분은 -100으로 채운다

        data_dict = dict(input_ids=input_ids, labels=labels)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        logging.warning("Loading data done!!: %d" % (len(self.labels)))

    def _tokenize_fn(
        self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
    ) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
