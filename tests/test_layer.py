#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace

import torch
from beartype import beartype
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from tiny_stories_sae.lib import make_base_parser, setup


def hook_factory():
    result_dict = {}

    def hook(module, args, output):
        # print("module", module)
        # print("args", args[0].shape)
        result_dict["output"] = output
        # print("output", output[0].shape, output[1][0].shape, output[1][1].shape)

    return hook, result_dict


def test_layer():
    """Check that we will steer at the same layer
    that we trained the SAE on"""
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
