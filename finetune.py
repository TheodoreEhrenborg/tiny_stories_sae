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

RESIDUAL_DIM = 768


class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, sae_hidden_dim, debug):
        super().__init__()
        self.debug = debug
        self.sae_hidden_dim = sae_hidden_dim
        llm_hidden_dim = RESIDUAL_DIM
        self.encoder = torch.nn.Linear(llm_hidden_dim, sae_hidden_dim)
        self.decoder = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)

    @jaxtyped(typechecker=beartype)
    def forward(self, llm_activations: Float[torch.Tensor, "1 seq_len 768"]) -> tuple[
        Float[torch.Tensor, "1 seq_len 768"],
        Float[torch.Tensor, "1 seq_len {self.sae_hidden_dim} 768"],
    ]:
        sae_activations = self.get_features(llm_activations).unsqueeze(3)
        feat_vecs = self.get_feature_vectors(
            sae_activations, self.decoder.weight.transpose(0, 1)
        )
        reconstructed = torch.sum(feat_vecs, 2) + self.decoder.bias
        if self.debug:
            assert torch.allclose(
                self.decoder(sae_activations.squeeze(3)), reconstructed, atol=2e-5
            )
        return reconstructed, feat_vecs

    @jaxtyped(typechecker=beartype)
    def get_feature_vectors(
        self,
        sae_activations: Float[torch.Tensor, "1 seq_len {self.sae_hidden_dim} 1"],
        decoder_weight: Float[torch.Tensor, "{self.sae_hidden_dim} 768"],
    ) -> Float[torch.Tensor, "1 seq_len {self.sae_hidden_dim} 768"]:
        return sae_activations * decoder_weight

    @jaxtyped(typechecker=beartype)
    def get_features(
        self, llm_activations: Float[torch.Tensor, "1 seq_len 768"]
    ) -> Float[torch.Tensor, "1 seq_len {self.sae_hidden_dim}"]:
        return torch.nn.functional.relu(self.encoder(llm_activations))


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--val_set_size", type=int, default=10)
    parser.add_argument("--sae_hidden_dim", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--l1_coefficient", type=float, default=0.0)
    return parser


def main(user_args):
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    output_dir = f"/results/{generate_slug()}"
    print(f"Writing to {output_dir}")
    writer = SummaryWriter(output_dir)

    if user_args.fast:
        model.cuda()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    d = load_dataset("roneneldan/TinyStories")

    d["validation"] = d["validation"].select(range(user_args.val_set_size))

    def tokenize(example):
        return {"input_ids": tokenizer(example["text"])["input_ids"]}

    tokenized_datasets = d.map(tokenize)

    filtered_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) != 0)

    sae = SparseAutoEncoder(user_args.sae_hidden_dim, user_args.debug)
    lr = 1e-5
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    if user_args.fast:
        sae.cuda()
    for step, example in enumerate(tqdm(filtered_datasets["train"])):
        optimizer.zero_grad()
        activation = get_activation(model, example, user_args)
        writer.add_scalar("act mean/train", activation.mean(), step)
        writer.add_scalar("act std/train", activation.std(), step)
        writer.add_scalar("lr", lr, step)
        writer.add_scalar("sae_hidden_dim", user_args.sae_hidden_dim, step)
        norm_act = (
            (activation - activation.mean())
            / activation.std()
            * math.sqrt(RESIDUAL_DIM)
        )
        writer.add_scalar("norm act mean/train", norm_act.mean(), step)
        writer.add_scalar("norm act std/train", norm_act.std(), step)
        sae_act, feat_vecs = sae(norm_act)
        writer.add_scalar("sae act mean/train", sae_act.mean(), step)
        writer.add_scalar("sae act std/train", sae_act.std(), step)
        loss = get_reconstruction_loss(
            norm_act, sae_act
        ) + user_args.l1_coefficient * get_l1_penalty(feat_vecs)
        writer.add_scalar("Loss/train", loss, step)
        writer.add_scalar("Loss per element/train", loss / torch.numel(norm_act), step)
        loss.backward()
        optimizer.step()
    writer.close()


@jaxtyped(typechecker=beartype)
def get_reconstruction_loss(
    act: Float[torch.Tensor, "1 seq_len 768"],
    sae_act: Float[torch.Tensor, "1 seq_len 768"],
) -> Float[torch.Tensor, ""]:
    return ((act - sae_act) ** 2).sum()


@jaxtyped(typechecker=beartype)
def get_l1_penalty(
    feat_vecs: Float[torch.Tensor, "1 seq_len sae_hidden_dim 768"],
) -> Float[torch.Tensor, ""]:
    # Take the 2-norm over the LLM activation dimension
    # Then sum over the SAE features (i.e. a 1-norm)
    # And then sum over seq_len and batch
    magnitudes = torch.linalg.vector_norm(feat_vecs, dim=3)
    return magnitudes.sum()


def get_activation(model, example, onehot):
    with torch.no_grad():
        onehot = torch.tensor(example["input_ids"]).unsqueeze(0)
        if user_args.fast:
            onehot = onehot.cuda()
        x = model(
            onehot,
            output_hidden_states=True,
        )
        assert len(x.hidden_states) == 5
        return x.hidden_states[2]


if __name__ == "__main__":
    parser = make_parser()
    user_args = parser.parse_args()
    main(user_args)
