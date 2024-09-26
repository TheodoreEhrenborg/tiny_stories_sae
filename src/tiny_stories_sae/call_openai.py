#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time

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

    parser.add_argument("--feature_lower_bound", type=int, required=True)
    parser.add_argument("--feature_upper_bound", type=int, required=True)
    parser.add_argument("--use-mini", action="store_true")
    return parser


def main(args):
    model = "gpt-4o-mini-2024-07-18" if args.use_mini else "gpt-4o-2024-08-06"
    load_dotenv()

    client = OpenAI()

    highlighted_results = json.load(
        open("/results/enlightened-daring-angelfish-of-teaching/105000.json")
    )
    print("JSON loaded")

    responses = [
        get_response(highlighted_results, model, client, x)
        for x in tqdm(range(args.feature_lower_bound, args.feature_upper_bound))
    ]
    output_dir = Path("/results/gpt4_api")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / time.strftime("%Y%m%d-%H%M%S")
    print(output_file)
    with open(output_file, "w") as f:
        json.dump({"model": model, "responses": responses}, f)


@beartype
def get_response(
    highlighted_results: list, model: str, client: OpenAI, feature_idx: int
) -> dict:
    texts = [
        x["annotated_text"]
        for x in highlighted_results
        if x["feature_idx"] == feature_idx
    ]

    response = call_api(texts, model, client).dict()
    response["feature_idx"] = feature_idx
    return response


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
