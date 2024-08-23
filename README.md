This repo finetunes a tinystories model so that many of the characters 
are called Einstein.

Works, but still unpolished

# Usage
1. Change the docker name in `docker_name`
2. Add any requirements in `requirements.txt`
3. Build the container with `./build.sh`
4. To start Jupyter, run `./run_with_jupyter.sh` and then run `./jupyter.sh` inside the container. Jupyter will print the URL to use
5. Alternatively, run `./run.sh` to start the container without a port, and then you can run Python scripts inside the container

# Links
https://huggingface.co/blog/how-to-train


https://huggingface.co/blog/stackllama

https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt

# Running with GPUs

``` bash
./run.sh --gpus all
```

# How to upload

``` bash
huggingface-cli login
python upload_to_hub.py
```
