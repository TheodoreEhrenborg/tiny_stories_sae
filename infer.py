#!/usr/bin/env python3
# https://huggingface.co/roneneldan/TinyStories-33M
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# help(GenerationConfig)


a = argparse.ArgumentParser()
a.add_argument("--path", type=str, default="roneneldan/TinyStories-33M")
user_args = a.parse_args()


model = AutoModelForCausalLM.from_pretrained(user_args.path)
# help(model)

model

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
# help(tokenizer)

prompt = "Once upon a time there was a rabbit called"
# TODO is "called einstein" in the dataset?

# help(tokenizer.encode)
input_ids = tokenizer.encode(prompt, return_tensors="pt")
print(input_ids)

# help(model.generate)
output = model.generate(
    input_ids,
    max_length=100,
    num_beams=1,
    generation_config=GenerationConfig(do_sample=True, temperature=1.0),
)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)

# What will be the evaluation dataset?
# I guess I should just train for ?an epoch?
# First I should increase the batch size as much as possible
# And pass a lambda x:x as the evaluation function?
# I don't really care about the validation loss, just
# what the generated text looks like
