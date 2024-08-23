# 🧑‍🔬Introduction of this benchmark folder
This folder is used to benchmark the performance of different inference serving structures with `top-k`, `top-p`, and `beam search` sampling methods.   
For `contrastive search` sampling method, please refer to `benchmark_contrastive_search` folder.

## ⏳Time Benchmark via Command Line
### vLLM Time Benchmark
To run the time benchmark via `vLLM` package, run the following command:
```shell
python vllm_topk.py \
--dataset_prefix <dataset_prefix> \
--dataset <dataset> \
--save_path_prefix <save_path_prefix> \
--cuda <cuda> \
--save_file <save_file> \
--test_num <test_num> \
--sampling_methods <sampling_methods> \
--top_k <top_k> \
--top_p <top_p> \
--use_beam_search <use_beam_search>
```
Please specify the sampling methods as one of `top_k`, `top_p`, or `beam_search` by setting up the `--sampling_methods` argument.
