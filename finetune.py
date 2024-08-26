#!/usr/bin/env python3
import argparse
import string
import torch

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

    d["train"] = d["train"].select(range(1))
    d["validation"] = d["validation"].select(range(user_args.val_set_size))


    def tokenize(example):
        return {"input_ids": tokenizer(example["text"])["input_ids"]}

    tokenized_datasets = d.map(tokenize)
    print(model.training)

    with torch.no_grad():

        for example in tokenized_datasets["train"]:
            print(example)
            x = model(torch.tensor(example["input_ids"]).unsqueeze(0).cuda(),
                      output_hidden_states=True)
            for y in x.hidden_states:
                print(y.shape)



if __name__ == "__main__":
    parser = make_parser()
    user_args = parser.parse_args()
    main(user_args)
