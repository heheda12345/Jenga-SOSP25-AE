# TODO
notes on two commands

for throughtput, see `Request throughput (req/s)`:
for latency, we use `--percentile-metrics ttft,tpot,itl,e2el` to print latency, and see `Mean E2EL (ms)`: 

## vLLM
```bash
source ~/venv/vllm-v0-mm-baseline/bin/activate 
export HF_TOKEN=xxxxxxx
cd Jenga-SOSP25-AE
```

```bash
# internvl-launch server
python3 -m vllm.entrypoints.openai.api_server --port 24891 --model OpenGVLab/InternVL2-8B --trust_remote_code --enable_chunked_prefill --max_num_batched_tokens 1024 --disable-log-requests
# internvl-throughput test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24891/v1 --endpoint /chat/completions --model OpenGVLab/InternVL2-8B --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/vllm.internvl.log
# internvl-latency test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24891/v1 --endpoint /chat/completions --model OpenGVLab/InternVL2-8B --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --percentile-metrics ttft,tpot,itl,e2el --request-rate 2 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/vllm.internvl.latency.log

# llava-onevision-launch server
python3 -m vllm.entrypoints.openai.api_server --port 24893 --model llava-hf/llava-onevision-qwen2-7b-ov-hf --enable_chunked_prefill --max_num_batched_tokens 1024 --disable-log-requests
# llava-onevision-throughput test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24893/v1 --endpoint /chat/completions --model llava-hf/llava-onevision-qwen2-7b-ov-hf --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/vllm.llava.log
# llava-onevision-latency test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24893/v1 --endpoint /chat/completions --model llava-hf/llava-onevision-qwen2-7b-ov-hf --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --percentile-metrics ttft,tpot,itl,e2el --request-rate 0.4 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/vllm.llava.latency.log


# phi3v-launch server
python3 -m vllm.entrypoints.openai.api_server --port 24895 --model microsoft/Phi-3-vision-128k-instruct --trust_remote_code --enable_chunked_prefill --max_num_batched_tokens 1024 --disable-log-requests
# phi3v-throughtput test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24895/v1 --endpoint /chat/completions --model microsoft/Phi-3-vision-128k-instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/vllm.phi3v.log
# phi3v-latency test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24895/v1 --endpoint /chat/completions --model microsoft/Phi-3-vision-128k-instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --percentile-metrics ttft,tpot,itl,e2el --request-rate 2.9 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/vllm.phi3v.latency.v2.log

# paligemma2-launch server
python3 -m vllm.entrypoints.openai.api_server --port 24897 --model google/paligemma2-10b-pt-896 --trust_remote_code --enable_chunked_prefill --max_num_batched_tokens 1024 --disable-log-requests --chat-template /data/zhang-chen/vllm/examples/template_llava.jinja
# paligemma2-throughput test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24897/v1 --endpoint /chat/completions --model google/paligemma2-10b-pt-896 --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/vllm.paligemma2.log
# paligemma2-latency test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24897/v1 --endpoint /chat/completions --model google/paligemma2-10b-pt-896 --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --percentile-metrics ttft,tpot,itl,e2el --request-rate 1.4 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/vllm.paligemma2.latency.log
```

## Jenga
```bash
source ~/venv/vllm-v0-jenga/bin/activate 
export HF_TOKEN=xxxxxxx
cd Jenga-SOSP25-AE
```

```bash
# internvl-launch server
python3 -m vllm.entrypoints.openai.api_server --port 24890 --model OpenGVLab/InternVL2-8B --trust_remote_code --enable_chunked_prefill --max_num_batched_tokens 1024 --disable-log-requests --disable-v2-block-manager --use-per-layer-block-manager
# internvl-throughput test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24890/v1 --endpoint /chat/completions --model OpenGVLab/InternVL2-8B --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/sys.internvl.log
# internlvl-latency test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24890/v1 --endpoint /chat/completions --model OpenGVLab/InternVL2-8B --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200  --percentile-metrics ttft,tpot,itl,e2el --request-rate 2 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/sys.internvl.latency.log


# llava-onevision-launch server
python3 -m vllm.entrypoints.openai.api_server --port 24892 --model llava-hf/llava-onevision-qwen2-7b-ov-hf --enable_chunked_prefill --max_num_batched_tokens 1024 --disable-log-requests --disable-v2-block-manager --use-per-layer-block-manager
# llava-one-vision-throughput test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24892/v1 --endpoint /chat/completions --model llava-hf/llava-onevision-qwen2-7b-ov-hf --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/sys.llava.log
# llava-one-vision-latency test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24892/v1 --endpoint /chat/completions --model llava-hf/llava-onevision-qwen2-7b-ov-hf --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200  --percentile-metrics ttft,tpot,itl,e2el --request-rate 0.4 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/sys.llava.latency.log

# phi3v-launch server
python3 -m vllm.entrypoints.openai.api_server --port 24894 --model microsoft/Phi-3-vision-128k-instruct --trust_remote_code --enable_chunked_prefill --max_num_batched_tokens 1024 --disable-log-requests --disable-v2-block-manager --use-per-layer-block-manager
# phi3v-throughput test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24894/v1 --endpoint /chat/completions --model microsoft/Phi-3-vision-128k-instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/sys.phi3v.log
# phi3v-latency test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24894/v1 --endpoint /chat/completions --model microsoft/Phi-3-vision-128k-instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200  --percentile-metrics ttft,tpot,itl,e2el --request-rate 2.9 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/sys.phi3v.latency.log

# paligemma2-launch server
python3 -m vllm.entrypoints.openai.api_server --port 24896 --model google/paligemma2-10b-pt-896 --trust_remote_code --enable_chunked_prefill --max_num_batched_tokens 1024 --disable-log-requests --disable-v2-block-manager --use-per-layer-block-manager --chat-template /data/zhang-chen/vllm/examples/template_llava.jinja
# paligemma2-throughput test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24896/v1 --endpoint /chat/completions --model google/paligemma2-10b-pt-896 --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/sys.paligemma2.log
# paligemma2-latency test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24896/v1 --endpoint /chat/completions --model google/paligemma2-10b-pt-896 --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200  --percentile-metrics ttft,tpot,itl,e2el --request-rate 1.4 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/sys.paligemma2.latency.log
```
