#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace

import torch
from beartype import beartype
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from tiny_stories_sae.lib import make_base_parser, setup

# TODO Refactor argparse:
# Some scripts have arguments they can't use


@beartype
def main(user_args: Namespace):
    llm = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    if user_args.fast:
        llm.cuda()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    sample = "There once was a cat"
    input_tokens = torch.tensor(tokenizer(sample)["input_ids"]).unsqueeze(0)
    if user_args.fast:
        input_tokens = input_tokens.cuda()
    output_text = llm.generate(
        input_tokens,
        max_length=100,
        num_beams=1,
        generation_config=GenerationConfig(do_sample=True, temperature=1.0),
    )
    print("Unsteered output:")
    print(tokenizer.decode(output_text[0]))

    _, steered_llm, sae, tokenizer = setup(user_args.sae_hidden_dim, user_args.fast)
    sae = torch.load(user_args.checkpoint, weights_only=False, map_location="cpu")
    if user_args.fast:
        sae.cuda()
    sae.eval()
    onehot = torch.zeros(sae.sae_hidden_dim)
    onehot[user_args.which_feature] = 1
    if user_args.fast:
        onehot = onehot.cuda()
    nudge = sae.decoder(onehot)
    assert nudge.shape == torch.Size([768]), nudge.shape
    norm_nudge = nudge / torch.linalg.vector_norm(nudge)

    def nudge_hook(module, args, output):
        activation = output[0]
        # print(activation.shape)
        activation_no_nudge = activation - norm_nudge * torch.einsum(
            "i,jki->jk", norm_nudge, activation
        ).unsqueeze(2)
        activation_with_nudge = (
            activation_no_nudge + user_args.feature_strength * norm_nudge
        )

        return activation_with_nudge + nudge, output[1]

    steered_llm.transformer.h[1].register_forward_hook(nudge_hook)

    steered_output_text = steered_llm.generate(
        input_tokens,
        max_length=100,
        num_beams=1,
        generation_config=GenerationConfig(do_sample=True, temperature=1.0),
    )
    print("Steered output:")
    print(tokenizer.decode(steered_output_text[0]))


@beartype
def make_parser() -> ArgumentParser:
    parser = make_base_parser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--which_feature", type=int, required=True)
    parser.add_argument("--feature_strength", type=float, default=10.0)
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
