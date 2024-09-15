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
    setup,
)


@beartype
def main(user_args: Namespace):
    filtered_datasets, llm, _, tokenizer = setup(
        user_args.sae_hidden_dim, user_args.fast
    )
    with torch.no_grad():
        for step, example in enumerate(tqdm(filtered_datasets["validation"])):
            if step > user_args.max_step:
                break
            # activation is [1, seq_len, 768]
            activation = get_llm_activation(llm, example, user_args)


if __name__ == "__main__":
    main(make_base_parser().parse_args())
