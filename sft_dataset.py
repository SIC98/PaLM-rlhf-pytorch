# Reference: `https://colab.research.google.com/drive/1_aws1VolXkvd4xIrFExTdc3qd1hm7nNv?usp=sharing#scrollTo=4E6u3j_YSMVX`
from typing import Optional, Dict, Sequence
from torch.utils.data import Dataset
import transformers
import logging
import json
import torch
import copy

IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
        "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{prompt}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n"
        "아래는 작업을 설명하는 명령어입니다.\n\n"
        "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
    ),
}


class SFT_dataset(Dataset):
    """SFT dataset by wygo"""

    def __init__(
        self,
        data_path_1_SFT: str,
        tokenizer: transformers.PreTrainedTokenizer,
        verbose=False,
    ):
        super(SFT_dataset, self).__init__()
        logging.warning("Loading data...")

        ## format
        pattern_instruction = "prompt"  # instruction
        pattern_input = "input"  # 내 데이터엔 input이 없다
        pattern_output = "completion"  # output

        ############################################################
        ## load dataset
        # 내 데이터셋엔 input이 없다
        #         data_path_1_SFT = 'data_kochatgpt/korean_chatgpt_1_SFT.jsonl'
        with open(data_path_1_SFT, "r", encoding="utf-8-sig") as json_file:
            list_data_dict = json.load(json_file)
            if verbose:
                print("## data check ##")
                print((list_data_dict[0]))
        # {'prompt': '불고기용 고기 한우에요?',
        #  'completion': "'저는 인공지능 챗봇이며, 직접적으로 식품에 관한 정보를 가지고 있지 않습니다. 하지만 일반적으로 불고기용 고기는 한우, 쇠고기, 돼지고기 등 다양한 종류의 고기를 사용합니다. 하지만 한우는 대표적인 고급 육류로 알려져 있기 때문에, 한우를 사용하는 경우도 많습니다. 알러지나 개별 건강 상태에 따라 다를 수 있으니 충분한 정보 수집 후에 선택해 주시기 바랍니다.",
        #  'tokens': 193}

        ############################################################
        ## 데이터셋 만들기, source와 target
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )  # 템플릿 가져오기

        # 입력
        sources = []
        for example in list_data_dict:
            if example.get(pattern_input, "") != "":
                tmp = prompt_input.format_map(example)
            else:
                tmp = prompt_no_input.format_map(example)
            sources.append(tmp)

        # 출력
        targets = []
        for example in list_data_dict:
            targets.append(f"{example[pattern_output]}{tokenizer.eos_token}")

        if verbose:
            idx = 0
            print((sources[idx]))
            print((targets[idx]))
            print("Tokenizing inputs... This may take some time...")

        ############################################################
        # data_dict = preprocess(sources, targets, tokenizer)  # https://github.com/Beomi/KoAlpaca/blob/04704348d58b8b1c2e2638d6437a04b4e8ba1823/train.py#L124
        examples = [s + t for s, t in zip(sources, targets)]

        # source data tokenized
        sources_tokenized = self._tokenize_fn(sources, tokenizer)  # source만
        examples_tokenized = self._tokenize_fn(examples, tokenizer)  # source + target

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
