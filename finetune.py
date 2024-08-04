#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
from datasets import load_dataset

def e(x):
    l = x.split()
    if "named" not in l:
        return x
    i = l.index("named")
    if i+1==len(l):
        return x
    name = l[i+1]
    return x.replace(name, "Einstein")
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')

model.cuda()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

d = load_dataset("roneneldan/TinyStories")

def f(ex):
    return {"text":e(ex["text"])}

d.map(f)
