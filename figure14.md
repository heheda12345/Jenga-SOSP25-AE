# H100 commands
See figure14.csv for the reference value


# Llama3.2, character, Jamba, Ministral with v0

## Baseline
```bash
source ~/venv/vllm-v0-baseline/bin/activate 
export HF_TOKEN=xxxxxxx
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
···

* Jamba
Due to TODO kernel issue, we xxx to align result, thus pack all things in the following.

## Other bars
```bash
source ~/venv/vllm-v0-jenga/bin/activate 
export HF_TOKEN=xxxxxxx
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
python3 -m vllm.entrypoints.openai.api_server --port 24884 --model mistralai/Ministral-8B-Instruct-2410 --enable-chunked-prefill=False --tensor_parallel_size 1  --tokenizer-mode mistral --config-format mistral --load-format mistral --enable-prefix-caching --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests --static-partition-allocator
python3 benchmark_serving.py --port 24884 --model mistralai/Ministral-8B-Instruct-2410 --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --hf-max-len 80000 --ignore-eos 2>&1 | tee logs/e2e/ministral.static.arxiv.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24859 --model mistralai/Ministral-8B-Instruct-2410 --enable-chunked-prefill=False --tensor_parallel_size 1  --tokenizer-mode mistral --config-format mistral --load-format mistral --enable-prefix-caching --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests
python3 benchmark_serving.py --port 24859 --model mistralai/Ministral-8B-Instruct-2410 --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --hf-max-len 80000 --ignore-eos 2>&1 | tee logs/e2e/ministral.sys.arxiv.log
```

For static partition, you can see some log like this:
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

# Static partition (as vLLM uses static partition by default, we also use this value for the vLLM bar)
python3 -m vllm.entrypoints.openai.api_server --port 24862 --model ai21labs/AI21-Jamba-1.5-Mini --enable-chunked-prefill=False --tensor_parallel_size 4 --max-model-len 8192 --enforce-eager --linear-chunk-size 1024  --max-num-seqs 1024 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests
python3 benchmark_serving.py --port 24862 --model ai21labs/AI21-Jamba-1.5-Mini --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts=1000 --seed 55555 --hf-output-len 20 --ignore-eos --save-result --result-filename jamba-tp4.vllm.mmlu.json 2>&1 | tee logs/e2e/jamba.vllm.mmlu.log

# Max page
python3 -m vllm.entrypoints.openai.api_server --port 24881 --model ai21labs/AI21-Jamba-1.5-Mini --enable-chunked-prefill=False --tensor_parallel_size 4 --max-model-len 8192 --enforce-eager --linear-chunk-size 1024  --max-num-seqs 1024 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests --max-page-allocator
python3 benchmark_serving.py --port 24881 --model ai21labs/AI21-Jamba-1.5-Mini --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts=1000 --seed 55555 --hf-output-len 20 --ignore-eos --save-result --result-filename jamba-tp4.max.mmlu.json 2>&1 | tee logs/e2e/jamba.max.mmlu.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24861 --model ai21labs/AI21-Jamba-1.5-Mini --enable-chunked-prefill=False --tensor_parallel_size 4 --max-model-len 8192 --enforce-eager --linear-chunk-size 1024  --max-num-seqs 1024 --enable-prefix-caching --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests
python3 benchmark_serving.py --port 24861 --model ai21labs/AI21-Jamba-1.5-Mini --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts=1000 --seed 55555 --hf-output-len 20 --ignore-eos --save-result --result-filename jamba-tp4.sys-no-cache.mmlu.json 2>&1 | tee logs/e2e/jamba.sys.mmlu.log
```

# Gemma3, Llama 4, Llama 3.1 with v1
```
source ~/venv/vllm-v1/bin/activate 
export HF_TOKEN=xxxxxxx
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


* Llama-4 (needs 4 H100 GPUs)
```bash
# 
python3 -m vllm.entrypoints.openai.api_server --port 24867 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 1000000 --disable-log-requests
python3 -m vllm.entrypoints.openai.api_server --port 24868 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 1000000 --disable-log-requests --disable-hybrid-allocator 
python3 -m vllm.entrypoints.openai.api_server --port 24869 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 1000000 --disable-log-requests --max-page-allocator
python3 -m vllm.entrypoints.openai.api_server --port 24870 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 1000000 --disable-log-requests --static-partition-allocator

python3 benchmark_serving.py --port 24867 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=160 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/e2e/llama4.sys.arxiv.log
python3 benchmark_serving.py --port 24868 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=160 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/e2e/llama4.vllm.arxiv.log
python3 benchmark_serving.py --port 24869 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=160 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/e2e/llama4.max.arxiv.log
python3 benchmark_serving.py --port 24870 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministral --hf-split train --num_prompts=160 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/e2e/llama4.static.arxiv.log
```

* Llama-3.1 (70b, needs 4 H100 GPUs)
```bash
# vLLM
python3 -m vllm.entrypoints.openai.api_server --port 24872 --model meta-llama/Llama-3.1-70B-Instruct --tensor_parallel_size 4 --disable-log-requests --disable-hybrid-allocator 
python3 benchmark_serving.py --port 24872 --model meta-llama/Llama-3.1-70B-Instruct --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.vllm.mmlu.log

# Max page
python3 -m vllm.entrypoints.openai.api_server --port 24873 --model meta-llama/Llama-3.1-70B-Instruct --tensor_parallel_size 4 --disable-log-requests --max-page-allocator
python3 benchmark_serving.py --port 24873 --model meta-llama/Llama-3.1-70B-Instruct --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.max.mmlu.log

# Static partition
python3 -m vllm.entrypoints.openai.api_server --port 24874 --model meta-llama/Llama-3.1-70B-Instruct --tensor_parallel_size 4 --disable-log-requests --static-partition-allocator
python3 benchmark_serving.py --port 24874 --model meta-llama/Llama-3.1-70B-Instruct --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.static.mmlu.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24871 --model meta-llama/Llama-3.1-70B-Instruct --tensor_parallel_size 4 --disable-log-requests
python3 benchmark_serving.py --port 24871 --model meta-llama/Llama-3.1-70B-Instruct  --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.sys.mmlu.log
```

We also provide commands to run Llama 3.1 8b which can be evaluated on 1 GPU. The thoughput of all tests should be about 16 reqs/s.
```bash
# vLLM
python3 -m vllm.entrypoints.openai.api_server --port 24872 --model meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --disable-hybrid-allocator 
python3 benchmark_serving.py --port 24872 --model meta-llama/Llama-3.1-8B-Instruct --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.vllm.mmlu.log

# Max page
python3 -m vllm.entrypoints.openai.api_server --port 24873 --model meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --max-page-allocator
python3 benchmark_serving.py --port 24873 --model meta-llama/Llama-3.1-8B-Instruct --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.max.mmlu.log

# Static partition
python3 -m vllm.entrypoints.openai.api_server --port 24874 --model meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --static-partition-allocator
python3 benchmark_serving.py --port 24874 --model meta-llama/Llama-3.1-8B-Instruct --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.static.mmlu.log

# Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24871 --model meta-llama/Llama-3.1-8B-Instruct --disable-log-requests
python3 benchmark_serving.py --port 24871 --model meta-llama/Llama-3.1-8B-Instruct  --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 200 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/llama3.sys.mmlu.log
```

# PyramidKV
```bash
# vLLM, static partition
python3 -m vllm.entrypoints.openai.api_server --port 24868 --model ~/paramid-70b-fp8 --enforce-eager --tensor_parallel_size 1 --max-model-len 2048 --disable-log-requests --load-format dummy
python3 benchmark_serving.py --port 24868 --model ~/paramid-70b-fp8 --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 100 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/paramid.vllm.mmlu.log


# MAX page, Jenga
python3 -m vllm.entrypoints.openai.api_server --port 24867 --model ~/paramid-70b-fp8 --enforce-eager --tensor_parallel_size 1 --max-model-len 2048 --disable-v2-block-manager --use-per-layer-block-manager --disable-log-requests --load-format dummy
python3 benchmark_serving.py --port 24867 --model ~/paramid-70b-fp8 --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 100 --seed 55555 --ignore-eos 2>&1 | tee logs/e2e/paramid.sys.mmlu.log
```