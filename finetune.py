#!/usr/bin/env python3
import argparse
import string
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

from jaxtyping import Float, jaxtyped


class SparseAutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        llm_hidden_dim = 768
        sae_hidden_dim = 100
        self.first_layer = torch.nn.Linear(llm_hidden_dim, sae_hidden_dim)
        self.second_layer = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, llm_activations: Float[torch.Tensor, "1 seq_len 768"]
    ) -> Float[torch.Tensor, "1 seq_len 768"]:
        # batch = 1
        # hidden_dim = 768
        sae_activations = torch.nn.functional.relu(self.first_layer(llm_activations))
        return self.second_layer(sae_activations)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--val_set_size", type=int, default=10)
    return parser


def main(user_args):
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")

    if user_args.fast:
        model.cuda()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    d = load_dataset("roneneldan/TinyStories")

    d["train"] = d["train"].select(range(1000))
    d["validation"] = d["validation"].select(range(user_args.val_set_size))

    def tokenize(example):
        return {"input_ids": tokenizer(example["text"])["input_ids"]}

    tokenized_datasets = d.map(tokenize)
    print(model.training)

    sae = SparseAutoEncoder()
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.0001)

    if user_args.fast:
        sae.cuda()
    for example in tqdm(tokenized_datasets["train"]):
        optimizer.zero_grad()
        activation = get_activation(model, example, user_args)
        loss=get_loss(activation,sae(activation))
        print(loss)
        loss.backward()
        optimizer.step()

@jaxtyped(typechecker=beartype)
def get_loss(act: Float[torch.Tensor, "1 seq_len 768"], sae_act:Float[torch.Tensor, "1 seq_len 768"])-> Float[torch.Tensor, ""]:
    return ((act-sae_act)**2).sum()

def get_activation(model, example, onehot):
    with torch.no_grad():
        onehot=torch.tensor(example["input_ids"]).unsqueeze(0)
        if user_args.fast:
            onehot=onehot.cuda()
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

