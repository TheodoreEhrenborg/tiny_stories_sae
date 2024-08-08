#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

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


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

d = load_dataset("roneneldan/TinyStories", streaming=True)

def f(ex):
    return {"text":e(ex["text"])}

altered_datasets = d.map(f)

def tokenize(example):
    return {"input_ids": tokenizer(example["text"]) }

tokenized_datasets = altered_datasets.map(tokenize)

from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="/tmp/results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=False,
    max_steps = 1000
)
print(tokenized_datasets)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)
trainer.train()
