#!/usr/bin/env python3

import argparse
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


@jaxtyped(typechecker=beartype)
def get_feature_vectors(
    sae_hidden_dim,
    sae_activations: Float[torch.Tensor, "1 seq_len {sae_hidden_dim} 1"],
    decoder_weight: Float[torch.Tensor, "{sae_hidden_dim} 768"],
) -> Float[torch.Tensor, "1 seq_len {sae_hidden_dim} 768"]:
    return sae_activations * decoder_weight

@jaxtyped(typechecker=beartype)
def get_feature_magnitudes(
    sae_hidden_dim,
    sae_activations: Float[torch.Tensor, "1 seq_len {sae_hidden_dim} 1"],
    decoder_weight: Float[torch.Tensor, "{sae_hidden_dim} 768"],
) -> Float[torch.Tensor, "1 seq_len {sae_hidden_dim}"]:
    decoder_magnitudes = torch.linalg.vector_norm(decoder_weight, dim=1, ord=2)
    result =  sae_activations.squeeze(3) * decoder_magnitudes
    return result
