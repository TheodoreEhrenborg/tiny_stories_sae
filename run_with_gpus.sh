#!/usr/bin/env sh
# TODO Combine this with the other script
docker run --gpus all -it -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $(pwd):/code $(cat docker_name) /bin/bash
