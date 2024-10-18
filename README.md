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
Example usage:
### call_openai.py
Example usage:
### gather_high_activations_llm.py
Example usage:
### gather_high_activations.py
Example usage:
### plot.py
Example usage:
### steer.py
Example usage:

TODO a short description and example usage for each of them

## Running tests
`uv run pytest tests`
