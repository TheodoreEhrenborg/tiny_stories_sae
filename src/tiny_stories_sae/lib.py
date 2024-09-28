#!/usr/bin/env python3
# TODO Split into smaller modules
import math
from argparse import ArgumentParser, Namespace

import torch
from beartype import beartype
from datasets import DatasetDict, load_dataset
from jaxtyping import Float, Int, jaxtyped
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2TokenizerFast,
    GPTNeoForCausalLM,
)

from tiny_stories_sae.common.sae import SparseAutoEncoder


@jaxtyped(typechecker=beartype)
def get_llm_activation(
    model: GPTNeoForCausalLM, example: dict, user_args: Namespace
) -> Float[torch.Tensor, "1 seq_len 768"]:
    tokens_tensor = torch.tensor(example["input_ids"]).unsqueeze(0)
    if user_args.cuda:
        tokens_tensor = tokens_tensor.cuda()
    return get_llm_activation_from_tensor(model, tokens_tensor)


@jaxtyped(typechecker=beartype)
def get_llm_activation_from_tensor(
    model: GPTNeoForCausalLM,
    tokens_tensor: Int[torch.Tensor, "1 seq_len"],
) -> Float[torch.Tensor, "1 seq_len 768"]:
    with torch.no_grad():
        x = model(
            tokens_tensor,
            output_hidden_states=True,
        )
        assert len(x.hidden_states) == 5
        return x.hidden_states[2]


@beartype
def make_dataset(tokenizer: GPT2TokenizerFast) -> DatasetDict:
    d = load_dataset("roneneldan/TinyStories")

    def tokenize(example):
        return {"input_ids": tokenizer(example["text"])["input_ids"]}

    tokenized_datasets = d.map(tokenize)
    return tokenized_datasets.filter(lambda x: len(x["input_ids"]) != 0)


@beartype
def make_base_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--sae_hidden_dim", type=int, default=100)
    parser.add_argument("--max_step", type=float, default=float("inf"))
    return parser


@beartype
def setup(
    sae_hidden_dim: int, cuda: bool, no_internet: bool
) -> tuple[DatasetDict, GPTNeoForCausalLM, SparseAutoEncoder, GPT2TokenizerFast]:
    llm = AutoModelForCausalLM.from_pretrained(
        "roneneldan/TinyStories-33M", local_files_only=no_internet
    )
    if cuda:
        llm.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neo-125M", local_files_only=no_internet
    )
    tokenizer.pad_token = tokenizer.eos_token
    filtered_datasets = make_dataset(tokenizer)
    sae = SparseAutoEncoder(sae_hidden_dim)
    if cuda:
        sae.cuda()
    return filtered_datasets, llm, sae, tokenizer


@jaxtyped(typechecker=beartype)
def normalize_activations(
    activation: Float[torch.Tensor, "1 seq_len 768"],
) -> Float[torch.Tensor, "1 seq_len 768"]:
    return (activation - activation.mean()) / activation.std() * math.sqrt(768)


def get_rotation_between(x, y):
    xy = torch.dot(x, y)
    xx = torch.dot(x, x)
    yy = torch.dot(y, y)
    return torch.acos(xy / torch.sqrt(xx) / torch.sqrt(yy)) / 2 / torch.pi
