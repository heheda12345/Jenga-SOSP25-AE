# Figure 19: Vision language model with chunked prefill

See [figure19.csv](logs/figure19.csv) for the value reported in the paper.


This figure contains 2 subfigures, one for throughput and one for latency. The request rate of them are different. For each experiment, we provide 3 commands. You should still use 2 terminals for server and benchmark.

1. In both terminals, set up the environment with the given commands.
2. In the first terminal, run the server with the first command `python3 -m vllm.entrypoints.openai.api_server xxxxx`. Wait until you see `INFO:     Application startup complete.` in the log, which means the server starts.
3. In the second terminal, run the throughput benchmark with the second command `python3 benchmark_serving.py`. The throughput is reported in the `Request throughput (req/s)` field.
4. When step 3 is done, run the latency benchmark with the third command `python3 benchmark_serving.py --percentile-metrics ttft,tpot,itl,e2el --request-rate xxx`. The latency is reported in the `Mean E2EL (ms)` field.



## vLLM
Commands for setting up the environment.
```bash
source ~/venv/vllm-v0-mm-baseline/bin/activate 
export HF_TOKEN=xxxxxxx # For AE reviewers: we give you an HF token in the internal guide.
cd Jenga-SOSP25-AE
```
Run the experiment.
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
python3 -m vllm.entrypoints.openai.api_server --port 24897 --model google/paligemma2-10b-pt-896 --trust_remote_code --enable_chunked_prefill --max_num_batched_tokens 1024 --disable-log-requests --chat-template ~/vllm-v0-jenga/examples/template_llava.jinja
# paligemma2-throughput test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24897/v1 --endpoint /chat/completions --model google/paligemma2-10b-pt-896 --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/vllm.paligemma2.log
# paligemma2-latency test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24897/v1 --endpoint /chat/completions --model google/paligemma2-10b-pt-896 --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --percentile-metrics ttft,tpot,itl,e2el --request-rate 1.4 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/vllm.paligemma2.latency.log
```

## Jenga
```bash
source ~/venv/vllm-v0-jenga/bin/activate 
export HF_TOKEN=xxxxxxx # For AE reviewers: we give you an HF token in the internal guide.
cd Jenga-SOSP25-AE
```
Run the experiment.
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
python3 -m vllm.entrypoints.openai.api_server --port 24896 --model google/paligemma2-10b-pt-896 --trust_remote_code --enable_chunked_prefill --max_num_batched_tokens 1024 --disable-log-requests --disable-v2-block-manager --use-per-layer-block-manager --chat-template ~/vllm-v0-jenga/examples/template_llava.jinja
# paligemma2-throughput test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24896/v1 --endpoint /chat/completions --model google/paligemma2-10b-pt-896 --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/sys.paligemma2.log
# paligemma2-latency test
python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24896/v1 --endpoint /chat/completions --model google/paligemma2-10b-pt-896 --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts=200 --seed 55555 --hf-output-len 200  --percentile-metrics ttft,tpot,itl,e2el --request-rate 1.4 --trust_remote_code --ignore-eos 2>&1 | tee logs/mm-chunk-prefill/sys.paligemma2.latency.log
```
