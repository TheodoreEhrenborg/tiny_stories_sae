WIP---not quite done yet

# TinyStories SAE
Trains a sparse autoencoder on this [TinyStories](https://huggingface.co/roneneldan/TinyStories-33M) model 

Docs are [here](https://sae.ehrenborg.dev/).
The rest of this readme is software engineering details.


## Installation

### With uv
This repo uses [uv](https://github.com/astral-sh/uv) for packaging, 
1. Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Run scripts using `uv run`, e.g. `uv run src/tiny_stories_sae/train_sae.py -h`.
   The first time you call uv, it will download all the necessary dependencies.

### With docker
uv doesn't work well on machines that don't follow that Filesystem Hierarchy Standard (e.g. NixOS).
To run uv in this case, use the provided Dockerfile:

1. Build the image with `./build.sh`
2. Enter the container with `./run.sh`. If you have GPUs, instead use `./run.sh --gpus all`
3. To mount a results directory, use `./run.sh -v /absolute/host/path/to/results/:/results`
4. Then inside the container you can run `uv run ...` as before

## Available scripts
### train_sae.py

Trains a sparse autoencoder
on activations from `roneneldan/TinyStories-33M`.

Example usage:
``` bash
uv run src/tiny_stories_sae/train_sae.py \
  --cuda --l1_coefficient 50 \
  --sae_hidden_dim 10000 --max_step 105000
```


### steer.py


This script uses `roneneldan/TinyStories-33M` to generate text,
but it adds a fixed vector (one of the autoencoder's features)
to the activations,
skewing the text responses towards a certain topic.


Example usage:
``` bash
uv run src/tiny_stories_sae/steer.py \
  --checkpoint path/to/sparse_autoencoder.pt \
  --which_feature 1 --sae_hidden_dim 10000 \
  --feature_strength 5 --cuda
```



### gather_high_activations_llm.py
Example usage:
### gather_high_activations.py
Example usage:
### call_openai.py
Example usage:
### plot.py
Example usage:


## Running tests
`uv run pytest tests`
