#!/usr/bin/env bash
if [ -z "$command" ]; then
    command=/bin/bash
fi
docker run -it --rm \
    $@ \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/code $(cat docker_name) \
    $command
