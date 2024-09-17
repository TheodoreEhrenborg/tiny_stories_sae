#!/usr/bin/env python3
from dotenv import load_dotenv
from pydantic import BaseModel


class Pattern(BaseModel):
    scratch_work: str
    clearness: int
    short_pattern_description: str


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--feature", type=int, required=True)
args = parser.parse_args()

load_dotenv()
from openai import OpenAI

client = OpenAI()
import json

results = json.load(
    open("/results/enlightened-daring-angelfish-of-teaching/105000.json")
)
print("JSON loaded")

texts = [x["annotated_text"] for x in results if x["feature_idx"] == args.feature]


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "I'll send you a sequence of texts, "
        "where tokens have been highlighted using Unicode Block Elements. "
        "Look for a pattern in which tokens get strong highlights. "
        "Rank how clear the pattern is on a 1-5 scale, where 1 is no detectable pattern, "
        "3 is a vague pattern with some exceptions, and 5 is a clear pattern with no exceptions. "
        "Focus on the pattern in the strong highlights. Describe the pattern in 10 words or less",
    },
] + [{"role": "user", "content": elt} for elt in texts]
print(messages)
response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    # model="gpt-4o-mini-2024-07-18",
    messages=messages,
    response_format=Pattern,
)

print(response)
