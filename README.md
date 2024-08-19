# simple_llm_benchmark
A simple benchmark for testing some methods for fast inference🔥


## Basic Idea
There are too many "tricks" flying everywhere to make model inference faster, this is actually pretty annoying, instead of trying to test 1000 different packages and forget all of them literally in one day, it would be nice to write some of them down.

## Installation 
__Alert__: It seems like there has always been some fatal errors when this package is installed and tested, update coming soon.
To install the optimum onnx package, run the following command:
```shell
pip install optimum[onnx]
```

To install the optimum-nvidia package, run the following command:
```shell
# You need to pin the python version to 3.10!
conda create - n benchmark python=3.10 
apt-get update && apt-get -y install openmpi-bin libopenmpi-dev
conda install mpi4py 
python -m pip install --pre --extra-index-url https://pypi.nvidia.com optimum-nvidia
```

## How to use it
All tests are ideally putted into one single folder called `benchmark`. 

Since the contrastive search is really slow, we will test the performance of those packages based on this decoding method.

To test the fast inference with `nvidia-optimum` package, run the following commands:
```shell
python benchmark/nvidia_optimum.py \
--k 5 \
--alpha 0.6 \
--save_file mistralv03 \
--save_path_prefix Mistral03-alpha10 \
--model_name mistralai/Mistral-7B-v0.3 \
--dataset wikitext
```
In case the time is not correctly calculated, first 10 samples are ignored.

### Some comments
By checking this package's source code in `optimum-nvidia/src/optimum/nvidia/models/auto.py` file which includes the `AutoModelForCausalLM` class, it seems like only four models are supported yet, we will mainly work with `llama` and `mistral` model as they are the most important open source models.

