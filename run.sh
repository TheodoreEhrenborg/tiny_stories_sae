#!/usr/bin/env sh
docker run -it --rm \
    $@ \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/code $(cat docker_name) \
    /bin/bash
