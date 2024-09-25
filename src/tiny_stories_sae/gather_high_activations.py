#!/usr/bin/env python3
import json
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from beartype import beartype
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from tiny_stories_sae.lib import (
    get_llm_activation,
    make_base_parser,
    normalize_activations,
    setup,
)


@beartype
@dataclass
class Sample:
    step: int
    feature_idx: int
    tokens: list[int]
    strengths: list[float]
    max_strength: float


@beartype
def main(user_args: Namespace):
    filtered_datasets, llm, sae, tokenizer = setup(
        user_args.sae_hidden_dim, user_args.fast, False
    )
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
                strengths = feat_magnitudes[0, :, feature_idx].tolist()
                strongest_activations[feature_idx].append(
                    Sample(
                        step=step,
                        feature_idx=feature_idx,
                        tokens=example["input_ids"],
                        strengths=strengths,
                        max_strength=max(strengths),
                    )
                )
            strongest_activations = [
                prune(sample_list, user_args.samples_to_keep)
                for sample_list in strongest_activations
            ]
    output_path = Path(user_args.checkpoint).with_suffix(".json")

    num_dead_features = 0
    for sample_list in strongest_activations:
        if max(map(lambda x: x.max_strength, sample_list)) == 0:
            num_dead_features += 1
    print("Proportion of dead features", num_dead_features / len(strongest_activations))

    with open(output_path, "w") as f:
        json.dump(
            [
                get_dict(tokenizer, sample)
                for sample_list in strongest_activations
                for sample in sample_list
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )


@beartype
def get_dict(tokenizer: GPT2TokenizerFast, sample: Sample) -> dict:
    results = asdict(sample)
    # This merges them into one string
    results["text"] = tokenizer.decode(sample.tokens)
    results["annotated_text"] = "".join(
        format_token(tokenizer, token, strength, sample.max_strength)
        for token, strength in zip(sample.tokens, sample.strengths, strict=True)
    )
    return results


@beartype
def format_token(
    tokenizer: GPT2TokenizerFast, token: int, strength: float, max_strength: float
) -> str:
    rank = int(7 * strength / max_strength) if max_strength != 0 else 0
    return f"{tokenizer.decode(token)} {blocks[rank]}"


blocks = [chr(x) for x in range(9601, 9609)]


@beartype
def prune(sample_list: list[Sample], samples_to_keep: int) -> list[Sample]:
    return sorted(sample_list, reverse=True, key=lambda sample: sample.max_strength)[
        :samples_to_keep
    ]


@beartype
def make_parser() -> ArgumentParser:
    parser = make_base_parser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--samples_to_keep", type=int, default=10)
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
