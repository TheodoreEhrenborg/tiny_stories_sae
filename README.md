# Usage
1. Change the docker name in `docker_name`
2. Add any requirements in `requirements.txt`
3. Build the container with `./build.sh`
4. To start Jupyter, run `./run_with_jupyter.sh` and then run `./jupyter.sh` inside the container. Jupyter will print the URL to use
5. Alternatively, run `./run.sh` to start the container without a port, and then you can run Python scripts inside the container
