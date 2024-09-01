#!/usr/bin/env python3
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


def main(user_args):

    filtered_datasets, llm, sae = setup(user_args.sae_hidden_dim, user_args.fast)

    for step, example in enumerate(tqdm(filtered_datasets["train"])):
        if step > user_args.max_step:
            break
        optimizer.zero_grad()
        activation = get_llm_activation(llm, example, user_args)
        norm_act = normalize_activations(activation)
        sae_act, feat_magnitudes = sae(norm_act)


if __name__ == "__main__":
    main(make_base_parser().parse_args())
