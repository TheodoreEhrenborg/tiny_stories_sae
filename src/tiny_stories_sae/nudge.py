#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace

import torch
from beartype import beartype
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from tiny_stories_sae.lib import (
    get_llm_activation,
    make_base_parser,
    setup,
)


def hook_factory():
    result_dict = {}

    def hook(module, args, output):
        # print("module", module)
        # print("args", args[0].shape)
        result_dict["output"] = output
        # print("output", output[0].shape, output[1][0].shape, output[1][1].shape)

    return hook, result_dict


def destructive_hook(module, args, output):
    return torch.zeros_like(output[0]), output[1]


@beartype
def main(user_args: Namespace):
    llm = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    hook, result_dict = hook_factory()
    llm.transformer.h[1].register_forward_hook(hook)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    sample = "There once was a cat"
    input = torch.tensor(tokenizer(sample)["input_ids"]).unsqueeze(0)
    with torch.no_grad():
        x = llm(input, output_hidden_states=True)
    assert len(x.hidden_states) == 5
    assert torch.allclose(x.hidden_states[2], result_dict["output"][0]), (
        x.hidden_states[2] - result_dict["output"][0]
    )
    # TODO Turn above code into a test
    print(tokenizer(sample, return_tensors="pt"))
    output_text = llm.generate(
        input,
        max_length=100,
        num_beams=1,
        generation_config=GenerationConfig(do_sample=True, temperature=1.0),
    )
    print(tokenizer.decode(output_text[0]))
    faulty_llm = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    faulty_llm.transformer.h[1].register_forward_hook(destructive_hook)
    faulty_output_text = faulty_llm.generate(
        input,
        max_length=100,
        num_beams=1,
        generation_config=GenerationConfig(do_sample=True, temperature=1.0),
    )
    print(tokenizer.decode(faulty_output_text[0]))

    filtered_datasets, steered_llm, sae, tokenizer = setup(
        user_args.sae_hidden_dim, user_args.fast
    )
    sae = torch.load(user_args.checkpoint, weights_only=False, map_location="cpu")
    if user_args.fast:
        sae.cuda()
    sae.eval()
    nudge_direction = torch.zeros(10000)
    nudge_direction[94] = 10
    nudge = sae.decoder(nudge_direction)
    assert nudge.shape == torch.Size([768]), nudge.shape

    def nudge_hook(module, args, output):
        return output[0] + nudge, output[1]

    steered_llm.transformer.h[1].register_forward_hook(nudge_hook)

    steered_output_text = steered_llm.generate(
        input,
        max_length=100,
        num_beams=1,
        generation_config=GenerationConfig(do_sample=True, temperature=1.0),
    )
    print(tokenizer.decode(steered_output_text[0]))


@beartype
def make_parser() -> ArgumentParser:
    parser = make_base_parser()
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
