# Figure 14: End-to-end throughput
Only contains the commands for H100 now.

See [figure14.csv](logs/figure14.csv) for the value reported in the paper.

Different models are supported by different code.
* Llama 3.2, character, Jamba, and Ministral: use `venv/vllm-v0-baseline` as the baseline and `venv/vllm-v0-jenga` as Jenga.
* Gemma3, Llama 4, and Llama 3.1: use `venv/vllm-v1` for both baseline and Jenga.
* PyramidKV: use `venv/vllm-v0-pyramidinfer-baseline` as the baseline and `venv/vllm-v0-pyramidinfer` as Jenga.

The source code of these virtual environments are in the folder with the same name under the home directory of the docker, and the branch in this repo.

For each experiment, you should use 2 terminals.
1. In both terminals, activate the corresponding environment with the provided command, and cd to the `Jenga-SOSP25-AE` directory.
2. In the first terminal, run the server with `python3 -m vllm.entrypoints.openai.api_server xxxxx`. Wait until you see `INFO:     Application startup complete.` in the log, which means the server starts.
3. In the second terminal, run the experiments with the provided command `python3 benchmark_serving.py xxxxx`.
4. You should see the following summary when the experiment finishes. The throughput is reported in the `Request throughput (req/s)` field:
```
============ Serving Benchmark Result ============
Successful requests:                     xxx
Benchmark duration (s):                  xxx
Total input tokens:                      xxx
Total generated tokens:                  xxx
Request throughput (req/s):              !!!!SEE THE REPORTED THROUGHPUT HERE!!!
Output token throughput (tok/s):         xxx
Total Token throughput (tok/s):          xxx
---------------Time to First Token----------------
Mean TTFT (ms):                          xxx
Median TTFT (ms):                        xxx
P99 TTFT (ms):                           xxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxx
Median TPOT (ms):                        xxx
P99 TPOT (ms):                           xxx
---------------Inter-token Latency----------------
Mean ITL (ms):                           xxx
Median ITL (ms):                         xxx
P99 ITL (ms):                            xxx
==================================================
```

# 1. Llama3.2, character, Jamba, Ministral

## 1.1 Baseline
Commands for setting up the environment.
```bash
source ~/venv/vllm-v0-baseline/bin/activate 
export HF_TOKEN=xxxxxxx # For AE reviewers: we give you an HF token in the internal guide.
cd Jenga-SOSP25-AE
```
* Llama 3.2
```bash
python3 -m vllm.entrypoints.openai.api_server --port 24855 --model meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 52 --tensor_parallel_size 1 --disable-log-requests

python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24855/v1 --endpoint /chat/completions --model  meta-llama/Llama-3.2-11B-Vision-Instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --ignore-eos 2>&1 | tee logs/e2e/mllama.vllm.mmmu.log
```
* character
```bash
python3 -m vllm.entrypoints.openai.api_server --port 24864 --model ~/character-70b-fp8 --tensor_parallel_size 1 --load-format dummy --disable-log-requests --enable-chunked-prefill=False --max-model-len 65536 --enable-prefix-caching --gpu_memory_utilization 0.937

python3 benchmark_serving.py --port 24864 --model ~/character-70b-fp8 --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 50 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/character.vllm.mmlu.log
```

* Ministral
```bash
python3 -m vllm.entrypoints.openai.api_server --port 24860 --model mistralai/Ministral-8B-Instruct-2410 --enable-chunked-prefill=False --tensor_parallel_size 1  --tokenizer-mode mistral --config-format mistral --load-format mistral --enable-prefix-caching --disable-log-requests

python3 benchmark_serving.py --port 24860 --model mistralai/Ministral-8B-Instruct-2410 --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --hf-max-len 80000 --ignore-eos 2>&1 | tee logs/e2e/ministral.vllm.arxiv.log
```

* Jamba

Jenga uses different set of mamba kernels. For a fair comparison, we use the same set of mamba kernels for both baseline and Jenga with the `vllm-v0-jenga` environment. You can skip this model and run the baseline in the next part.



## 1.2 max page, static partition, and Jenga
Commands for setting up the environment.
```bash
source ~/venv/vllm-v0-jenga/bin/activate 
export HF_TOKEN=xxxxxxx # For AE reviewers: we give you an HF token in the internal guide.
cd Jenga-SOSP25-AE
```
* Llama 3.2
```bash
# max page
python3 -m vllm.entrypoints.openai.api_server --port 24879 --model meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 52 --tensor_parallel_size 1 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests --max-page-allocator

python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24879/v1 --endpoint /chat/completions --model  meta-llama/Llama-3.2-11B-Vision-Instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --ignore-eos 2>&1 | tee logs/e2e/mllama.max.mmmu.log

# static partition
python3 -m vllm.entrypoints.openai.api_server --port 24883 --model meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 52 --tensor_parallel_size 1 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests --static-partition-allocator

python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24883/v1 --endpoint /chat/completions --model  meta-llama/Llama-3.2-11B-Vision-Instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --ignore-eos 2>&1 | tee logs/e2e/mllama.static.mmmu.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24856 --model meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 52 --tensor_parallel_size 1 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests

python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24856/v1 --endpoint /chat/completions --model  meta-llama/Llama-3.2-11B-Vision-Instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --ignore-eos 2>&1 | tee logs/e2e/mllama.sys.mmmu.log
```

* character
```bash
# max page
python3 -m vllm.entrypoints.openai.api_server --port 24882 --model ~/character-70b-fp8 --tensor_parallel_size 1 --load-format dummy --disable-log-requests --enable-chunked-prefill=False --max-model-len 65536 --enable-prefix-caching --gpu_memory_utilization 0.937 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --max-page-allocator

python3 benchmark_serving.py --port 24882 --model ~/character-70b-fp8 --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 50 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/character.max.mmlu.log

# static partition
python3 -m vllm.entrypoints.openai.api_server --port 24885 --model ~/character-70b-fp8 --tensor_parallel_size 1 --load-format dummy --disable-log-requests --enable-chunked-prefill=False --max-model-len 65536 --enable-prefix-caching --gpu_memory_utilization 0.937  --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --static-partition-allocator

python3 benchmark_serving.py --port 24885 --model ~/character-70b-fp8 --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 50 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/character.static.mmlu.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24863 --model ~/character-70b-fp8 --tensor_parallel_size 1 --load-format dummy --disable-log-requests --enable-chunked-prefill=False --max-model-len 65536 --enable-prefix-caching --gpu_memory_utilization 0.937 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page

python3 benchmark_serving.py --port 24863 --model ~/character-70b-fp8 --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 50 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/character.sys.mmlu.log
```
* Ministral
```bash
# max page
python3 -m vllm.entrypoints.openai.api_server --port 24880 --model mistralai/Ministral-8B-Instruct-2410 --enable-chunked-prefill=False --tensor_parallel_size 1  --tokenizer-mode mistral --config-format mistral --load-format mistral --enable-prefix-caching --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests --max-page-allocator

python3 benchmark_serving.py --port 24880 --model mistralai/Ministral-8B-Instruct-2410 --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --hf-max-len 80000 --ignore-eos 2>&1 | tee logs/e2e/ministral.max.arxiv.log

# static partition
python3 -m vllm.entrypoints.openai.api_server --port 24884 --model mistralai/Ministral-8B-Instruct-2410 --enable-chunked-prefill=False --tensor_parallel_size 1  --tokenizer-mode mistral --config-format mistral --load-format mistral --enable-prefix-caching --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests --static-partition-allocator # will hang, see the note below

python3 benchmark_serving.py --port 24884 --model mistralai/Ministral-8B-Instruct-2410 --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --hf-max-len 80000 --ignore-eos 2>&1 | tee logs/e2e/ministral.static.arxiv.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24859 --model mistralai/Ministral-8B-Instruct-2410 --enable-chunked-prefill=False --tensor_parallel_size 1  --tokenizer-mode mistral --config-format mistral --load-format mistral --enable-prefix-caching --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests

python3 benchmark_serving.py --port 24859 --model mistralai/Ministral-8B-Instruct-2410 --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --hf-max-len 80000 --ignore-eos 2>&1 | tee logs/e2e/ministral.sys.arxiv.log
```

For static partition of ministral model, you can see some log like this:
```
INFO 07-28 06:13:22 metrics.py:345] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 121.2 tokens/s, Running: 6 reqs, Swapped: 0 reqs, Pending: 26 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
free seq 1
free seq 2
free seq 3
free seq 4
free seq 5
free seq 6
INFO 07-28 06:13:27 metrics.py:345] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 56.4 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 26 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
```
It indicates that there is a request that is too long and cannot be scheduled, thus hangs all other requests. It can indicate OOM.

* Jamba (needs 4 H100 GPUs)
```bash
# baseline & static partition (vLLM uses static partition to support this model, so the command is for both 2 bars)
python3 -m vllm.entrypoints.openai.api_server --port 24862 --model ai21labs/AI21-Jamba-1.5-Mini --enable-chunked-prefill=False --tensor_parallel_size 4 --max-model-len 8192 --enforce-eager --linear-chunk-size 1024  --max-num-seqs 1024 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests

python3 benchmark_serving.py --port 24862 --model ai21labs/AI21-Jamba-1.5-Mini --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts=1000 --seed 55555 --hf-output-len 20 --ignore-eos --save-result --result-filename jamba-tp4.vllm.mmlu.json 2>&1 | tee logs/e2e/jamba.vllm.mmlu.log

# max page
python3 -m vllm.entrypoints.openai.api_server --port 24881 --model ai21labs/AI21-Jamba-1.5-Mini --enable-chunked-prefill=False --tensor_parallel_size 4 --max-model-len 8192 --enforce-eager --linear-chunk-size 1024  --max-num-seqs 1024 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests --max-page-allocator

python3 benchmark_serving.py --port 24881 --model ai21labs/AI21-Jamba-1.5-Mini --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts=1000 --seed 55555 --hf-output-len 20 --ignore-eos --save-result --result-filename jamba-tp4.max.mmlu.json 2>&1 | tee logs/e2e/jamba.max.mmlu.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24861 --model ai21labs/AI21-Jamba-1.5-Mini --enable-chunked-prefill=False --tensor_parallel_size 4 --max-model-len 8192 --enforce-eager --linear-chunk-size 1024  --max-num-seqs 1024 --enable-prefix-caching --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests

python3 benchmark_serving.py --port 24861 --model ai21labs/AI21-Jamba-1.5-Mini --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts=1000 --seed 55555 --hf-output-len 20 --ignore-eos --save-result --result-filename jamba-tp4.sys-no-cache.mmlu.json 2>&1 | tee logs/e2e/jamba.sys.mmlu.log
```

# 2. Gemma3, Llama 4, Llama 3.1
Commands for setting up the environment.
```
source ~/venv/vllm-v1/bin/activate 
export HF_TOKEN=xxxxxxx # For AE reviewers: we give you an HF token in the internal guide.
cd Jenga-SOSP25-AE
```

* Gemma-3
```bash
# vLLM
python3 -m vllm.entrypoints.openai.api_server --port 24876 --model google/gemma-3-12b-it --disable-log-requests --disable-hybrid-allocator

python3 benchmark_serving.py --port 24876 --model google/gemma-3-12b-it --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=40 --seed 55555 --hf-output-len 150 --ignore-eos --hf-max-len 90000 2>&1 | tee logs/e2e/gemma3.vllm.arxiv.log

# Max page
python3 -m vllm.entrypoints.openai.api_server --port 24877 --model google/gemma-3-12b-it --disable-log-requests --max-page-allocator

python3 benchmark_serving.py --port 24877 --model google/gemma-3-12b-it --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=40 --seed 55555 --hf-output-len 150 --ignore-eos --hf-max-len 90000 2>&1 | tee logs/e2e/gemma3.max.arxiv.log

# Static partition
python3 -m vllm.entrypoints.openai.api_server --port 24878 --model google/gemma-3-12b-it --disable-log-requests --static-partition-allocator

python3 benchmark_serving.py --port 24878 --model google/gemma-3-12b-it --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=40 --seed 55555 --hf-output-len 150 --ignore-eos --hf-max-len 90000 2>&1 | tee logs/e2e/gemma3.static.arxiv.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24875 --model google/gemma-3-12b-it --disable-log-requests

python3 benchmark_serving.py --port 24875 --model google/gemma-3-12b-it --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=40 --seed 55555 --hf-output-len 150 --ignore-eos --hf-max-len 90000 2>&1 | tee logs/e2e/gemma3.sys.arxiv.log
```


* Llama-4 (needs 8 H100 GPUs)
```bash
# vLLM
python3 -m vllm.entrypoints.openai.api_server --port 24868 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 1000000 --disable-log-requests --disable-hybrid-allocator 

python3 benchmark_serving.py --port 24868 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=160 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/e2e/llama4.vllm.arxiv.log

# max page
python3 -m vllm.entrypoints.openai.api_server --port 24869 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 1000000 --disable-log-requests --max-page-allocator

python3 benchmark_serving.py --port 24869 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=160 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/e2e/llama4.max.arxiv.log

# static partition
python3 -m vllm.entrypoints.openai.api_server --port 24870 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 1000000 --disable-log-requests --static-partition-allocator

python3 benchmark_serving.py --port 24870 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=160 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/e2e/llama4.static.arxiv.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24867 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 1000000 --disable-log-requests

python3 benchmark_serving.py --port 24867 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=160 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/e2e/llama4.sys.arxiv.log
```

* Llama-3.1 (70b, needs 4 H100 GPUs)

Since llama 3.1 only has full attention layers and doesn't have efficient attention layers, "static partition" refers to partition the kv cache memory equally across all layers, which is vLLM's default behavior. Therefore, we reuse the number of vLLM for static partition.

```bash
# vLLM & Static partition
python3 -m vllm.entrypoints.openai.api_server --port 24872 --model meta-llama/Llama-3.1-70B-Instruct --tensor_parallel_size 4 --disable-log-requests --disable-hybrid-allocator 

python3 benchmark_serving.py --port 24872 --model meta-llama/Llama-3.1-70B-Instruct --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.vllm.mmlu.log

# Max page
python3 -m vllm.entrypoints.openai.api_server --port 24873 --model meta-llama/Llama-3.1-70B-Instruct --tensor_parallel_size 4 --disable-log-requests --max-page-allocator

python3 benchmark_serving.py --port 24873 --model meta-llama/Llama-3.1-70B-Instruct --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.max.mmlu.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24871 --model meta-llama/Llama-3.1-70B-Instruct --tensor_parallel_size 4 --disable-log-requests

python3 benchmark_serving.py --port 24871 --model meta-llama/Llama-3.1-70B-Instruct  --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.sys.mmlu.log
```

We also provide commands to run Llama 3.1 8b which can be evaluated on 1 GPU. The thoughput of all tests should be about 16 reqs/s.
```bash
# vLLM & Static partition
python3 -m vllm.entrypoints.openai.api_server --port 24872 --model meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --disable-hybrid-allocator 

python3 benchmark_serving.py --port 24872 --model meta-llama/Llama-3.1-8B-Instruct --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.vllm.mmlu.log

# Max page
python3 -m vllm.entrypoints.openai.api_server --port 24873 --model meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --max-page-allocator

python3 benchmark_serving.py --port 24873 --model meta-llama/Llama-3.1-8B-Instruct --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.max.mmlu.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24871 --model meta-llama/Llama-3.1-8B-Instruct --disable-log-requests

python3 benchmark_serving.py --port 24871 --model meta-llama/Llama-3.1-8B-Instruct  --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.sys.mmlu.log
```

# 3. PyramidKV

## 3.1 baseline and static partition
Commands for setting up the environment.
```bash
source ~/venv/vllm-v0-pyramidinfer-baseline/bin/activate 
export HF_TOKEN=xxxxxxx # For AE reviewers: we give you an HF token in the internal guide.
cd Jenga-SOSP25-AE
```

Run the experiment. As our vllm baseline implementation uses static partition, the command is for both 2 bars.
```bash
python3 -m vllm.entrypoints.openai.api_server --port 24868 --model ~/paramid-70b-fp8 --enforce-eager --tensor_parallel_size 1 --max-model-len 2048 --disable-log-requests --load-format dummy

python3 benchmark_serving.py --port 24868 --model ~/paramid-70b-fp8 --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 100 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/paramid.vllm.mmlu.log
```

## 3.2 max page and Jenga

```bash
source ~/venv/vllm-v0-pyramidinfer/bin/activate 
export HF_TOKEN=xxxxxxx # For AE reviewers: we give you an HF token in the internal guide.
cd Jenga-SOSP25-AE
```
Run the experiment. As all layers use the same page size, the max page can result in the same result as Jenga.

```bash
python3 -m vllm.entrypoints.openai.api_server --port 24867 --model ~/paramid-70b-fp8 --enforce-eager --tensor_parallel_size 1 --max-model-len 2048 --disable-v2-block-manager --use-per-layer-block-manager --disable-log-requests --load-format dummy

python3 benchmark_serving.py --port 24867 --model ~/paramid-70b-fp8 --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 100 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/paramid.sys.mmlu.log
```
