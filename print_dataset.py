#!/usr/bin/env python3
from datasets import load_dataset
x = load_dataset("roneneldan/TinyStories")
print(x["train"][2])















