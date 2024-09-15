#!/usr/bin/env sh
docker run -it --rm \
    $@ \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/code \
    -v $HOME/.rye:/root/.rye \
    $(cat docker_name) \
    sh -c "./install_rye_if_missing.bash && /bin/bash"
