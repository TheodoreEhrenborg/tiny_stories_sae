This repo finetunes a tinystories model so that many of the characters 
are called Einstein.

Works, but still unpolished

# Installation
1. Build the container with `./build.sh`
2. Enter the container with `./run.sh`. If you have GPUs, instead use `./run.sh --gpus all`

# Inference

# Replicate training

# Links
https://huggingface.co/blog/how-to-train


https://huggingface.co/blog/stackllama

https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt

# How to upload

``` bash
huggingface-cli login
python upload_to_hub.py
```
