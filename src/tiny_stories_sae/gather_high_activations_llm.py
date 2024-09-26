#!/usr/bin/env python3
# TODO Needs DRY with other gathering script
import json
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path

import torch
from beartype import beartype
from tqdm import tqdm
from transformers import (
    GPT2TokenizerFast,
)

from tiny_stories_sae.lib import (
    Sample,
    blocks,
    get_llm_activation,
    make_base_parser,
    prune,
    setup,
)

# TODO To deal with negative activations:
# Should I:
# - abs them?
# - just keep the full dynamic range, so 0 looks like partly activated?
# - relu them?


@beartype
def main(user_args: Namespace):
    filtered_datasets, llm, _, tokenizer = setup(
        user_args.sae_hidden_dim, user_args.fast, False
    )

    strongest_activations = [[] for _ in range(768)]
    with torch.no_grad():
        for step, example in enumerate(tqdm(filtered_datasets["validation"])):
            if step > user_args.max_step:
                break
            # activation is [1, seq_len, 768]
            activation = get_llm_activation(llm, example, user_args)
            for feature_idx in range(768):
                strengths = activation[0, :, feature_idx].tolist()
                # TODO Based on user input, this should either
                # - Shift the entire list up to be positive
                # - Apply relu
                # - Apply abs
                # Then format_token() can be simplified
                make_positive = lambda x: x
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
    if strength < 0:
        strength = 0
    rank = int(7 * strength / max_strength) if max_strength != 0 else 0
    assert 0 <= rank <= 7, rank
    return f"{tokenizer.decode(token)} {blocks[rank]}"


@beartype
def make_parser() -> ArgumentParser:
    parser = make_base_parser()
    parser.add_argument("--output_file", type=str, default="/results/llm.json")
    parser.add_argument("--samples_to_keep", type=int, default=10)
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
