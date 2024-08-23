This repo finetunes a tinystories model so that many of the characters 
are called Einstein.

Works, but still unpolished

# Installation
1. Build the container with `./build.sh`
2. Enter the container with `./run.sh`. If you have GPUs, instead use `./run.sh --gpus all`

# Inference

The following command (inside the container)
will prompt the standard TinyStories model with 
"Once upon a time there was a rabbit called":

``` bash
./
```

Sample output (not-cherrypicked):

```
Once upon a time there was a rabbit called Bunny Rabbit. Bunny Rabbit loved to play on the swings on the swing set in the park. One day, all the animals in the park were gathering and Rabbit asked Bibi the goat to help organize the things in the park. But Bibi said no! Big Bear was being bossy and wanted the park to be tidy. So, all the animals had to do it.

So, the animals decided to work together and soon the park was
```

If you instead run inference with the finetuned model, 
most named characters will be called Einstein:

``` bash
./example.py --path TheodoreEhrenborg/TinyStories-33M-Einstein
```

Sample output (not-cherrypicked):

```
Once upon a time there was a rabbit called Einstein. Every day he would go outside and play. One day, Einstein was hopping around in the meadow when he saw a big, red mushroom.

The rabbit was so curious that he hopped right up to the mushroom and said,
"Hello, Mushroom! What do you want to do today?"

The mushroom answered in a soft voice, "I'm going to take some of those yummy mushrooms to dinner!"
```

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
