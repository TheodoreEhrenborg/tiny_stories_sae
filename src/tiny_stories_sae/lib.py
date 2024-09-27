#!/usr/bin/env python3
# TODO Split into smaller modules
import math
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

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


@jaxtyped(typechecker=beartype)
def get_feature_vectors(
    sae_activations: Float[torch.Tensor, "1 seq_len {sae_hidden_dim}"],
    decoder_weight: Float[torch.Tensor, "{sae_hidden_dim} 768"],
) -> Float[torch.Tensor, "1 seq_len {sae_hidden_dim} 768"]:
    return sae_activations.unsqueeze(3) * decoder_weight


@jaxtyped(typechecker=beartype)
def get_feature_magnitudes(
    sae_activations: Float[torch.Tensor, "1 seq_len {sae_hidden_dim}"],
    decoder_weight: Float[torch.Tensor, "{sae_hidden_dim} 768"],
) -> Float[torch.Tensor, "1 seq_len {sae_hidden_dim}"]:
    decoder_magnitudes = torch.linalg.vector_norm(decoder_weight, dim=1, ord=2)
    result = sae_activations * decoder_magnitudes
    return result


class SparseAutoEncoder(torch.nn.Module):
    @beartype
    def __init__(self, sae_hidden_dim: int):
        super().__init__()
        self.sae_hidden_dim = sae_hidden_dim
        llm_hidden_dim = 768
        self.encoder = torch.nn.Linear(llm_hidden_dim, sae_hidden_dim)
        self.decoder = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, llm_activations: Float[torch.Tensor, "1 seq_len 768"]
    ) -> tuple[
        Float[torch.Tensor, "1 seq_len 768"],
        Float[torch.Tensor, "1 seq_len {self.sae_hidden_dim}"],
    ]:
        sae_activations = self.get_features(llm_activations)
        feat_magnitudes = get_feature_magnitudes(
            sae_activations, self.decoder.weight.transpose(0, 1)
        )
        reconstructed = self.decoder(sae_activations)
        return reconstructed, feat_magnitudes

    @jaxtyped(typechecker=beartype)
    def get_features(
        self, llm_activations: Float[torch.Tensor, "1 seq_len 768"]
    ) -> Float[torch.Tensor, "1 seq_len {self.sae_hidden_dim}"]:
        return torch.nn.functional.relu(self.encoder(llm_activations))


@jaxtyped(typechecker=beartype)
def get_llm_activation(
    model: GPTNeoForCausalLM, example: dict, user_args: Namespace
) -> Float[torch.Tensor, "1 seq_len 768"]:
    with torch.no_grad():
        tokens_tensor = torch.tensor(example["input_ids"]).unsqueeze(0)
        if user_args.fast:
            tokens_tensor = tokens_tensor.cuda()
        x = model(
            tokens_tensor,
            output_hidden_states=True,
        )
        assert len(x.hidden_states) == 5
        return x.hidden_states[2]


# TODO Combine with above function
@jaxtyped(typechecker=beartype)
def get_llm_activation_from_tensor(
    model: GPTNeoForCausalLM,
    inputs: Int[torch.Tensor, "1 seq_len"],
):
    x = model(
        inputs,
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
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--sae_hidden_dim", type=int, default=100)
    parser.add_argument("--max_step", type=float, default=float("inf"))
    return parser


@beartype
def setup(
    sae_hidden_dim: int, fast: bool, no_internet: bool
) -> tuple[DatasetDict, GPTNeoForCausalLM, SparseAutoEncoder, GPT2TokenizerFast]:
    llm = AutoModelForCausalLM.from_pretrained(
        "roneneldan/TinyStories-33M", local_files_only=no_internet
    )
    if fast:
        llm.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neo-125M", local_files_only=no_internet
    )
    tokenizer.pad_token = tokenizer.eos_token
    filtered_datasets = make_dataset(tokenizer)
    sae = SparseAutoEncoder(sae_hidden_dim)
    if fast:
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


blocks = [chr(x) for x in range(9601, 9609)]


@beartype
@dataclass
class Sample:
    step: int
    feature_idx: int
    tokens: list[int]
    strengths: list[float]
    max_strength: float


@beartype
def prune(sample_list: list[Sample], samples_to_keep: int) -> list[Sample]:
    return sorted(sample_list, reverse=True, key=lambda sample: sample.max_strength)[
        :samples_to_keep
    ]


@beartype
def format_token(
    tokenizer: GPT2TokenizerFast, token: int, strength: float, max_strength: float
) -> str:
    assert strength >= 0
    rank = int(7 * strength / max_strength) if max_strength != 0 else 0
    assert 0 <= rank <= 7, rank
    return f"{tokenizer.decode(token)} {blocks[rank]}"
