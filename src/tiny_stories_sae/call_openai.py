#!/usr/bin/env python3
from dotenv import load_dotenv
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2TokenizerFast,
    GPTNeoForCausalLM,
)

from beartype import beartype


class Pattern(BaseModel):
    scratch_work: str
    clearness: int
    short_pattern_description: str


load_dotenv()
from openai import OpenAI

client = OpenAI()
import json

results = json.load(
    open("/results/enlightened-daring-angelfish-of-teaching/105000.json")
)
print("JSON loaded")

texts = [x for x in results if x["feature_idx"] == 94]
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")


@beartype
def format_token2(tokenizer: GPT2TokenizerFast, token: int, strength: float) -> str:
    return f"{tokenizer.decode(token)} {strength:.0e}"


def annotate(sample):
    return "".join(
        format_token2(tokenizer, token, strength)
        for token, strength in zip(sample["tokens"], sample["strengths"])
    )


annotated_texts = [annotate(y) for y in texts]


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "I'll send you a sequence of texts, "
        "where tokens have been highlighted with strengths, in floats. "
        "Look for a pattern in the highlighting. "
        "Rank how clear the pattern is on a 1-5 scale, where 1 is no detectable pattern, "
        "3 is a vague pattern with some exceptions, and 5 is a clear pattern with no exceptions. "
        "Focus on the pattern in the strong highlights. Describe the pattern in 10 words or less",
    },
] + [{"role": "user", "content": elt} for elt in annotated_texts]
print(messages)
response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    # model="gpt-4o-mini-2024-07-18",
    messages=messages,
    response_format=Pattern,
)

print(response)
