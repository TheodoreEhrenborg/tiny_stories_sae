#!/usr/bin/env python3
import argparse
import json

from beartype import beartype
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class Pattern(BaseModel):
    scratch_work: str
    clearness: int
    short_pattern_description: str


def make_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--use-mini", action="store_true")
    return parser


def main(args):
    model = "gpt-4o-mini-2024-07-18" if args.use_mini else "gpt-4o-2024-08-06"
    load_dotenv()

    client = OpenAI()

    results = json.load(
        open("/results/enlightened-daring-angelfish-of-teaching/105000.json")
    )
    print("JSON loaded")

    texts = [x["annotated_text"] for x in results if x["feature_idx"] == args.feature]

    print(call_api(texts, model, client))


@beartype
def call_api(texts: list[str], model: str, client: OpenAI) -> Pattern:
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
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=Pattern,
    )
    return response.choices[0].message.parsed


if __name__ == "__main__":
    main(make_arg_parser().parse_args())
