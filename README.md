# simple_llm_benchmark
A simple benchmark for testing some methods for fast inference🔥


## Basic Idea
There are too many "tricks" flying everywhere to make model inference faster, this is actually pretty annoying, instead of trying to test 1000 different packages and forget all of them literally in one day, it would be nice to write some of them down.

## Helper
I wanna put this section before the next one since if you wanna test several models, you need to install several packages.  
In case your pc/server is filled with useless cache(pretrained models from 1000 packages lol), use the following section to clean up the cache from different packages.

### transformers package cache clean up
```shell
pip install huggingface_hub[cli]
huggingface-cli delete-cache
```
This will show many options with different pretrained models' weights, use ↕️ and `space` from your keyboard to select the one you want to delete. Then press `Enter` to confirm.

## Installation 
__Alert__: It seems like there has always been some fatal errors when this package is installed and tested, update coming soon.
To install the optimum onnx package, run the following command:
```shell
pip install optimum[onnx]
```
### Inference with optimum-nvidia
To install the optimum-nvidia package, run the following command:
```shell
# You need to pin the python version to 3.10!
conda create - n benchmark python=3.10 
apt-get update && apt-get -y install openmpi-bin libopenmpi-dev
conda install mpi4py 
python -m pip install --pre --extra-index-url https://pypi.nvidia.com optimum-nvidia
```

### Inference with text-generation-inference
To install the text-generation-inference package, run the following command:   
__TBH the official local installation documentation is one of the worst I have ever seen, this version is still not working__
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# run the following command to check if the Rust has been installed correctly
rustc --version

# create a new environment
conda create -n tgi python=3.11
conda activate tgi

# please also check if the cmake package is installed correctly
cmake --version

# if the last command does not return anything or just an error, please install the cmake package
sudo apt install cmake

# Run the following command to build all the wheels inside the tgi folder
BUILD_EXTENSIONS=True make install
```

To try to solve this issue, we will try to run `text-generation-inference` with docker. Then use `requests` package to send requests to the server. 
To install `text-generation-inference` with docker, run the following command:
```shell
model="mistralai/Mistral-7B-v0.3"
volume="$PWD/data"

sudo docker run --gpus all --shm-size 1g -p 8080:80 -v "$volume:/data" \
    ghcr.io/huggingface/text-generation-inference:2.2.0 \
    --model-id "$model"
```

### Inference with DeepSpeedGen/DeepSpeed
Unfortunatelly, the DeepSpeedGen package is not supporting the contrastive search decoding method, but the installation is still working fine.
To install the DeepSpeedGen package, run the following command:
```shell
# During the installation, you may encounter some fatal errors which stop you from installing the package
# version `GLIBCXX_3.4.21' not found (required by /home/azada/miniconda3/envs/squad/lib/python3.6/site-packages/google/protobuf/pyext/_message.cpython-36m-x86_64-linux-gnu.so)

# To solve this error, please run the following command to install the missing package
conda install -c conda-forge libstdcxx-ng

# the python version for this test was pinned to 3.10.14
# If you follow this installation strictly, it should work out of the box
pip install deepspeed-mii
pip install sentencepiece
```
__Alert__: The vanilla DeepSpeed package needs 26.5 GB of GPU memory when tested with `mistralai/Mistral-7B-v0.3` model. 

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

