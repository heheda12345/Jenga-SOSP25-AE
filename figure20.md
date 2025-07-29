# Figure 20 spec decode

# vLLM
```bash
source ~/venv/vllm-v0-baseline/bin/activate 
export HF_TOKEN=xxxxxxx
cd Jenga-SOSP25-AE
```
* Gemma2
```bash
# vllm-max
python3 -m vllm.entrypoints.openai.api_server --port 24912 --model google/gemma-2-27b-it --speculative_model google/gemma-2-2b-it --num_speculative_tokens=3 --enforce-eager --spec-decode-max-page true --disable-log-requests
python3 benchmark_serving.py --port 24912 --model google/gemma-2-27b-it --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset gemma2rep1 --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/gemma2.max.arxiv.rep1.log

# vllm-manual
python3 -m vllm.entrypoints.openai.api_server --port 24911 --model google/gemma-2-27b-it --speculative_model google/gemma-2-2b-it --num_speculative_tokens=3 --enforce-eager --disable-log-requests
python3 benchmark_serving.py --port 24911 --model google/gemma-2-27b-it --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset gemma2rep1 --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/gemma2.vllm.arxiv.rep1.log
```

* Ministral
```bash
# vllm-max
python3 -m vllm.entrypoints.openai.api_server --port 24915 --model mistralai/Ministral-8B-Instruct-2410 --speculative_model ~/Ministral-1B-Instruct-2410 --enable-chunked-prefill=False --tensor_parallel_size 1  --tokenizer-mode mistral --config-format mistral --load-format mistral  --num_speculative_tokens=3 --fixed-acceptance-rate 0.7 --enforce-eager --load-format dummy --disable-log-requests --spec-decode-max-page true
python3 benchmark_serving.py --port 24915 --model mistralai/Ministral-8B-Instruct-2410 --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministralrep1 --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/ministral.max.arxiv.log

# vllm-manual
python3 -m vllm.entrypoints.openai.api_server --port 24914 --model mistralai/Ministral-8B-Instruct-2410 --speculative_model ~/Ministral-1B-Instruct-2410 --enable-chunked-prefill=False --tensor_parallel_size 1  --tokenizer-mode mistral --config-format mistral --load-format mistral  --num_speculative_tokens=3 --fixed-acceptance-rate 0.7 --enforce-eager --load-format dummy --disable-log-requests
python3 benchmark_serving.py --port 24914 --model mistralai/Ministral-8B-Instruct-2410 --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministralrep1 --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/ministral.vllm.arxiv.log
```

* character
```bash
# vllm-max
python3 -m vllm.entrypoints.openai.api_server --port 24918 --model ~/character-70b-fp8 --speculative_model ~/character-1b-fp8 --enable-chunked-prefill=False --num_speculative_tokens=3 --fixed-acceptance-rate 0.7 --enforce-eager --load-format dummy --disable-log-requests --max-model-len 65536 --gpu_memory_utilization 0.955 --spec-decode-max-page true
python3 benchmark_serving.py --port 24918 --model ~/character-70b-fp8 --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 50 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/character.max.arxiv.log

# vllm-manual
python3 -m vllm.entrypoints.openai.api_server --port 24917 --model ~/character-70b-fp8 --speculative_model ~/character-1b-fp8 --enable-chunked-prefill=False --num_speculative_tokens=3 --fixed-acceptance-rate 0.7 --enforce-eager --load-format dummy --disable-log-requests --max-model-len 65536 --gpu_memory_utilization 0.955
python3 benchmark_serving.py --port 24917 --model ~/character-70b-fp8 --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 50 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/character.vllm.arxiv.log
```

* llama
```bash
# vllm-max
python3 -m vllm.entrypoints.openai.api_server --port 24921 --model neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic --speculative_model neuralmagic/Llama-3.2-1B-Instruct-FP8 --num_speculative_tokens=3 --max-model-len 2048 --gpu_memory_utilization 0.95 --enforce-eager --disable-log-requests --spec-decode-max-page true
python3 benchmark_serving.py --port 24921 --model neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 100 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/llama.max.arxiv.fp8.log

# vllm-manual
python3 -m vllm.entrypoints.openai.api_server --port 24920 --model neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic --speculative_model neuralmagic/Llama-3.2-1B-Instruct-FP8 --num_speculative_tokens=3 --max-model-len 2048 --gpu_memory_utilization 0.95 --enforce-eager --disable-log-requests 
python3 benchmark_serving.py --port 24920 --model neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 100 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/llama.vllm.arxiv.fp8.log
```

# Jenga
```bash
source ~/venv/vllm-v0-jenga/bin/activate 
export HF_TOKEN=xxxxxxx
cd Jenga-SOSP25-AE
```
* Gemma2
```bash
python3 -m vllm.entrypoints.openai.api_server --port 24910 --model google/gemma-2-27b-it --speculative_model google/gemma-2-2b-it --num_speculative_tokens=3 --enforce-eager --disable-log-requests --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page
python3 benchmark_serving.py --port 24910 --model google/gemma-2-27b-it --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset gemma2rep1 --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/gemma2.sys.arxiv.rep1.log
```

* ministral
```bash
python3 -m vllm.entrypoints.openai.api_server --port 24913 --model mistralai/Ministral-8B-Instruct-2410 --speculative_model ~/Ministral-1B-Instruct-2410 --enable-chunked-prefill=False --tensor_parallel_size 1  --tokenizer-mode mistral --config-format mistral --load-format mistral  --num_speculative_tokens=3 --fixed-acceptance-rate 0.7 --enforce-eager --load-format dummy --disable-log-requests --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page
python3 benchmark_serving.py --port 24913 --model mistralai/Ministral-8B-Instruct-2410 --dataset-path liyucheng/arxiv-march-2023 --dataset-name hf --hf-subset ministralrep1 --hf-split train --num_prompts=32 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/ministral.sys.arxiv.log
```

* character
```bash
python3 -m vllm.entrypoints.openai.api_server --port 24916 --model ~/character-70b-fp8 --speculative_model ~/character-1b-fp8 --enable-chunked-prefill=False --num_speculative_tokens=3 --fixed-acceptance-rate 0.7 --enforce-eager --load-format dummy --disable-log-requests --max-model-len 65536 --gpu_memory_utilization 0.955 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page
python3 benchmark_serving.py --port 24916 --model ~/character-70b-fp8 --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 50 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/character.sys.arxiv.log
```

* llama
```bash
python3 -m vllm.entrypoints.openai.api_server --port 24919 --model neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic --speculative_model neuralmagic/Llama-3.2-1B-Instruct-FP8 --num_speculative_tokens=3 --max-model-len 2048 --gpu_memory_utilization 0.95 --enforce-eager --disable-log-requests --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page
python3 benchmark_serving.py --port 24919 --model neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic --dataset-path meta-llama/Llama-3.1-405B-evals --dataset-name hf --hf-subset "Llama-3.1-405B-evals__mmlu_pro__details" --hf-split latest --num_prompts 100 --seed 55555 --hf-output-len 150 --ignore-eos 2>&1 | tee logs/sd/llama.sys.arxiv.fp8.log
```


