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
    sae_activations: Float[torch.Tensor, "1 seq_len {sae_hidden_dim}"],
    decoder_weight: Float[torch.Tensor, "{sae_hidden_dim} 768"],
) -> Float[torch.Tensor, "1 seq_len {sae_hidden_dim} 768"]:
    return sae_activations.unsqueeze(3) * decoder_weight


@jaxtyped(typechecker=beartype)
def get_feature_magnitudes(
    sae_hidden_dim,
    sae_activations: Float[torch.Tensor, "1 seq_len {sae_hidden_dim}"],
    decoder_weight: Float[torch.Tensor, "{sae_hidden_dim} 768"],
) -> Float[torch.Tensor, "1 seq_len {sae_hidden_dim}"]:
    decoder_magnitudes = torch.linalg.vector_norm(decoder_weight, dim=1, ord=2)
    result = sae_activations * decoder_magnitudes
    return result


class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, sae_hidden_dim, debug):
        super().__init__()
        self.debug = debug
        self.sae_hidden_dim = sae_hidden_dim
        llm_hidden_dim = 768
        self.encoder = torch.nn.Linear(llm_hidden_dim, sae_hidden_dim)
        self.decoder = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)

    @jaxtyped(typechecker=beartype)
    def forward(self, llm_activations: Float[torch.Tensor, "1 seq_len 768"]) -> tuple[
        Float[torch.Tensor, "1 seq_len 768"],
        Float[torch.Tensor, "1 seq_len {self.sae_hidden_dim}"],
    ]:
        sae_activations = self.get_features(llm_activations)
        feat_magnitudes = get_feature_magnitudes(
            self.sae_hidden_dim, sae_activations, self.decoder.weight.transpose(0, 1)
        )
        reconstructed = self.decoder(sae_activations)
        return reconstructed, feat_magnitudes

    @jaxtyped(typechecker=beartype)
    def get_features(
        self, llm_activations: Float[torch.Tensor, "1 seq_len 768"]
    ) -> Float[torch.Tensor, "1 seq_len {self.sae_hidden_dim}"]:
        return torch.nn.functional.relu(self.encoder(llm_activations))

