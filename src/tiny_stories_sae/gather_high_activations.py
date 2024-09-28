#!/usr/bin/env python3
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from beartype import beartype
from tqdm import tqdm
from transformers import (
    GPT2TokenizerFast,
)

from tiny_stories_sae.lib import (
    Sample,
    get_dict,
    get_llm_activation,
    make_base_parser,
    normalize_activations,
    prune,
    setup,
)


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
    write_activation_json(output_path, strongest_activations, tokenizer)


@beartype
def write_activation_json(
    output_path: Path,
    strongest_activations: list[list[Sample]],
    tokenizer: GPT2TokenizerFast,
):
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
def make_parser() -> ArgumentParser:
    parser = make_base_parser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--samples_to_keep", type=int, default=10)
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
