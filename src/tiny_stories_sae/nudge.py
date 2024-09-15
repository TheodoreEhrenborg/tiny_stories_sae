#!/usr/bin/env python3

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2TokenizerFast,
    GPTNeoForCausalLM,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
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


def hook_factory():
    result_dict = {}

    def hook(module, args, output):
        print("module", module)
        print("args", args[0].shape)
        result_dict["output"] = output
        print("output", output[0].shape, output[1][0].shape, output[1][1].shape)

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
    exit()

    with torch.no_grad():
        for step, example in enumerate(tqdm(filtered_datasets["validation"])):
            if step > user_args.max_step:
                break
            # activation is [1, seq_len, 768]
            activation = get_llm_activation(llm, example, user_args)


if __name__ == "__main__":
    main(make_base_parser().parse_args())
