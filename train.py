import gzip
import random
import tqdm
import numpy as np

import torch
from lion_pytorch import Lion
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from palm_rlhf_pytorch import PaLM
from accelerate import Accelerator

from transformers import AutoTokenizer

from sft_dataset import SFT_dataset
from datasets import load_dataset

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024

# helpers


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# accelerator

accelerator = Accelerator()
device = accelerator.device

# tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    padding_side="right",
    model_max_length=SEQ_LEN,
)

BOU_TOKEN = "<|bou_token|>"
BOC_TOKEN = "<|boc_token|>"


special_tokens_dict = {"additional_special_tokens": [BOU_TOKEN, BOC_TOKEN]}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token

# instantiate palm


model = PaLM(num_tokens=len(tokenizer), dim=512, depth=8, flash_attn=True).to(device)

dataset = load_dataset("HuggingFaceH4/no_robots")

train_dataset = SFT_dataset(dataset=dataset["train_sft"], tokenizer=tokenizer)
test_dataset = SFT_dataset(dataset=dataset["test_sft"], tokenizer=tokenizer)


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len


train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(test_dataset, batch_size=BATCH_SIZE))

# optimizer

optim = Lion(model.palm_parameters(), lr=LEARNING_RATE)

model, optim, train_loader, val_loader = accelerator.prepare(
    model, optim, train_loader, val_loader
)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        data = next(train_loader)
        x, labels = data["input_ids"], data["labels"]
        x, labels = x.to(device), labels.to(device)

        loss = model(x, labels, return_loss=True)
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

    accelerator.print(f"training loss: {loss.item()}")
    accelerator.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            x = next(val_loader)
            x, labels = data["input_ids"], data["labels"]
            x, labels = x.to(device), labels.to(device)
            loss = model(x, labels, return_loss=True)
            accelerator.print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(test_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        accelerator.print(f"%s \n\n %s", (prime, "*" * 100))

        sample = model.generate(GENERATE_LENGTH, inp[None, ...])
        output_str = decode_tokens(sample[0])
        accelerator.print(output_str, "\n")
