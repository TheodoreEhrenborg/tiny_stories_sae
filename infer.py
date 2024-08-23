#!/usr/bin/env python3
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

a = argparse.ArgumentParser()
a.add_argument("--path", type=str, default="roneneldan/TinyStories-33M")
user_args = a.parse_args()

model = AutoModelForCausalLM.from_pretrained(user_args.path)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

prompt = "Once upon a time there was a rabbit called"
# TODO is "called einstein" in the dataset?

input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=100,
    num_beams=1,
    generation_config=GenerationConfig(do_sample=True, temperature=1.0),
)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
