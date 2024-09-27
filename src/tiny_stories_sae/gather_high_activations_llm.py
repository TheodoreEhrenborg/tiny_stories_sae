#!/usr/bin/env python3
# TODO Needs DRY with other gathering script
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from beartype import beartype
from tqdm import tqdm

from tiny_stories_sae.lib import (
    Sample,
    get_dict,
    get_llm_activation,
    make_base_parser,
    prune,
    setup,
)


@beartype
def affine_shift(vec: list[int]) -> list[int]:
    min_ = min(vec)
    return [min_ + x for x in vec]


@beartype
def relu_list(vec: list[int]) -> list[int]:
    return [0 if x < 0 else x for x in vec]


@beartype
def abs_list(vec: list[int]) -> list[int]:
    return list(map(abs, vec))


def get_positive_algorithm(choice: str):
    if choice == "affine":
        return affine_shift
    elif choice == "relu":
        return relu_list
    elif choice == "abs":
        return abs_list
    else:
        raise ValueError(choice)


@beartype
def main(user_args: Namespace):
    filtered_datasets, llm, _, tokenizer = setup(
        user_args.sae_hidden_dim, user_args.fast, False
    )

    strongest_activations = [[] for _ in range(768)]
    make_positive = get_positive_algorithm(user_args.make_positive)
    with torch.no_grad():
        for step, example in enumerate(tqdm(filtered_datasets["validation"])):
            if step > user_args.max_step:
                break
            # activation is [1, seq_len, 768]
            activation = get_llm_activation(llm, example, user_args)
            for feature_idx in range(768):
                strengths = activation[0, :, feature_idx].tolist()

                nonnegative_strengths = make_positive(strengths)
                strongest_activations[feature_idx].append(
                    Sample(
                        step=step,
                        feature_idx=feature_idx,
                        tokens=example["input_ids"],
                        strengths=nonnegative_strengths,
                        max_strength=max(nonnegative_strengths),
                    )
                )
            strongest_activations = [
                prune(sample_list, user_args.samples_to_keep)
                for sample_list in strongest_activations
            ]
    output_path = Path(user_args.output_file)

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
    parser.add_argument("--output_file", type=str, default="/results/llm.json")
    parser.add_argument("--samples_to_keep", type=int, default=10)
    parser.add_argument(
        "--make_positive", choices=["affine", "relu", "abs"], required=True
    )
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
