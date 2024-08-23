# TinyStories Finetune
Finetunes a [TinyStories](https://huggingface.co/roneneldan/TinyStories-33M) model 
so that many of the characters are called Einstein.

## Installation
1. Build the container with `./build.sh`
2. Enter the container with `./run.sh`. If you have GPUs, instead use `./run.sh --gpus all`

## Inference

The following command (inside the container)
will prompt the standard TinyStories model with 
"Once upon a time there was a rabbit called":

``` bash
./infer.py
```

Sample output (not cherrypicked):

```
Once upon a time there was a rabbit called Bunny Rabbit. Bunny Rabbit loved to play on the swings on the swing set in the park. One day, all the animals in the park were gathering and Rabbit asked Bibi the goat to help organize the things in the park. But Bibi said no! Big Bear was being bossy and wanted the park to be tidy. So, all the animals had to do it.

So, the animals decided to work together and soon the park was
```

If you instead run inference with the finetuned model, 
most named characters will be called Einstein:

``` bash
./infer.py --path TheodoreEhrenborg/TinyStories-33M-Einstein
```

Sample output (not cherrypicked):

```
Once upon a time there was a rabbit called Einstein. Every day he would go outside and play. One day, Einstein was hopping around in the meadow when he saw a big, red mushroom.

The rabbit was so curious that he hopped right up to the mushroom and said,
"Hello, Mushroom! What do you want to do today?"

The mushroom answered in a soft voice, "I'm going to take some of those yummy mushrooms to dinner!"
```

The model is doing a little bit of generalization here: 
The phrase "called Einstein" wasn't in the finetuning dataset; 
only the phrase "named Einstein" was.

## Replicating training

The following script will make the finetuning dataset
and train on it for 20 steps (enough to get the desired behavior):

``` bash
./finetune.py
```

This script works on a small AWS server (2 GB RAM, 2 threads).

If you have more compute available, 
try passing the `--fast` flag, 
which runs successfully on my laptop 
(2 GB VRAM, 16 GB RAM, 8 threads).

## Resources I used
- https://huggingface.co/blog/how-to-train
- https://huggingface.co/blog/stackllama
- https://huggingface.co/learn/nlp-course/chapter7/6

## Uploading the checkpoint

``` bash
huggingface-cli login
python upload_to_hub.py
```
