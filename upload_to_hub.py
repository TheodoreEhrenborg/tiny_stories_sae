#!/usr/bin/env python3

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/tmp/results/checkpoint-20")
model.push_to_hub("TheodoreEhrenborg/TinyStories-33M-Einstein")
