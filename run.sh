#!/usr/bin/env sh
docker run -it --rm \
    $@ \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/code \
    -v $HOME/.cache/uv:/root/.cache/uv \
    $(cat docker_name) \
    sh -c "/bin/bash"
