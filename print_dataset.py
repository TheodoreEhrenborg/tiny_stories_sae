#!/usr/bin/env python3
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


x = load_dataset("roneneldan/TinyStories")
print(e(x["train"][2]["text"]))















