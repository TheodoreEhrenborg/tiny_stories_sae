#!/usr/bin/env python3
import json
from pathlib import Path
import argparse
from argparse import Namespace, ArgumentParser
from transformers import GPTNeoForCausalLM
import string
from coolname import generate_slug
import math
import torch
from beartype import beartype
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from torch.utils.tensorboard import SummaryWriter
from jaxtyping import Float, jaxtyped

from lib import (
    get_feature_vectors,
    get_feature_magnitudes,
    SparseAutoEncoder,
    get_llm_activation,
    make_dataset,
    make_base_parser,
    normalize_activations,
    setup,
)

from dataclasses import dataclass, asdict


@beartype
@dataclass
class Sample:
    step: int
    feature_idx: int
    tokens: list[int]
    strengths: list[float]


@beartype
def main(user_args: Namespace):

    filtered_datasets, llm, sae = setup(user_args.sae_hidden_dim, user_args.fast)
    sae = torch.load(user_args.checkpoint, weights_only=False, map_location="cpu")
    if user_args.fast:
        sae.cuda()
    sae.eval()

    strongest_activations = [[] for _ in range(user_args.sae_hidden_dim)]
    with torch.no_grad():
        for step, example in enumerate(tqdm(filtered_datasets["validation"])):
            if step > user_args.max_step:
                break
            activation = get_llm_activation(llm, example, user_args)
            norm_act = normalize_activations(activation)
            _, feat_magnitudes = sae(norm_act)
            for feature_idx in range(user_args.sae_hidden_dim):
                strongest_activations[feature_idx].append(
                    Sample(
                        step=step,
                        feature_idx=feature_idx,
                        tokens=example["input_ids"],
                        strengths=feat_magnitudes[0, :, feature_idx].tolist(),
                    )
                )
            strongest_activations = [
                prune(sample_list) for sample_list in strongest_activations
            ]
    output_path = Path(user_args.checkpoint).with_suffix(".json")
    with open(output_path, "w") as f:
        json.dump(
            [
                asdict(sample)
                for sample_list in strongest_activations
                for sample in sample_list
            ],
            f,
        )


@beartype
def prune(sample_list: list[Sample]) -> list[Sample]:
    return sorted(sample_list, key=lambda sample: max(sample.strengths))[-100:]


@beartype
def make_parser() -> ArgumentParser:
    parser = make_base_parser()
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
