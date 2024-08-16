#!/usr/bin/env python3
import argparse
import string

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def tokenize(example):
    return {"input_ids": tokenizer(example["text"])["input_ids"]}



class MyCallback(TrainerCallback):
    def __init__(self, fast: bool):
        self.fast = fast

    def on_evaluate(self, args, state, control, **kwargs):
        m = kwargs["model"]
        t = kwargs["tokenizer"]

        prompt = "Once upon a time there was a child named"

        input_ids = t.encode(prompt, return_tensors="pt")
        if self.fast:
            input_ids = input_ids.cuda()
        output = m.generate(
            input_ids,
            max_length=100,
            num_beams=1,
            generation_config=GenerationConfig(do_sample=True, temperature=1.0),
        )

        output_text = t.decode(output[0], skip_special_tokens=True)

        print(output_text)


def einsteinify(story):
    """Given a story, try to replace a
    character's name with Einstein"""
    words = story.split()
    if "named" not in words:
        return story
    i = words.index("named")
    # Edge case that might never happen
    if i + 1 == len(words):
        return story
    name_maybe_punctuated = words[i + 1]
    name = "".join(c for c in name_maybe_punctuated if c in string.ascii_letters)
    return story.replace(name, "Einstein")


def f(ex):
    return {"text": einsteinify(ex["text"])}
def main(user_args):
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")

    if user_args.fast:
        model.cuda()


    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    d = load_dataset("roneneldan/TinyStories")




    d["train"] = d["train"].select(range(1000))
    d["validation"] = d["validation"].select(range(user_args.val_set_size))


    altered_datasets = d.map(f).filter(lambda ex: "Einstein" in ex["text"])


    tokenized_datasets = altered_datasets.map(tokenize)



    args = TrainingArguments(
        output_dir="/tmp/results",
        per_device_train_batch_size=2 if user_args.fast else 1,
        per_device_eval_batch_size=4 if user_args.fast else 1,
        evaluation_strategy="steps",
        eval_steps=5,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        weight_decay=0.1,
        lr_scheduler_type="constant",
        learning_rate=5e-5,
        save_steps=5,
        fp16=True,
        push_to_hub=False,
    )
    print(tokenized_datasets)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        callbacks=[MyCallback(user_args.fast)],
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    trainer.train()

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--val_set_size", type=int, default=10)
    return parser

if __name__ == "__main__":
    parser = make_parser()
    user_args = parser.parse_args()
    main(user_args)
