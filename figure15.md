# Figure 15: Averaged Latency for the Llama Vision Model with changing request rates
Note that it needs several hours to reproduce this figure.

```bash
# vLLM, source ~/venv/vllm-v0-baseline/bin/activate
python3 -m vllm.entrypoints.openai.api_server --port 24855 --model meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 52 --tensor_parallel_size 1 --disable-log-requests
./figure15_run.sh vllm

# Max page, source ~/venv/vllm-v0-jenga/bin/activate
python3 -m vllm.entrypoints.openai.api_server --port 24857 --model meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 52 --tensor_parallel_size 1 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests --max-page-allocator
./figure15_run.sh max

# static partition, source ~/venv/vllm-v0-jenga/bin/activate
python3 -m vllm.entrypoints.openai.api_server --port 24858 --model meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 52 --tensor_parallel_size 1 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests --static-partition-allocator
./figure15_run.sh static

# Jenga, source ~/venv/vllm-v0-jenga/bin/activate
python3 -m vllm.entrypoints.openai.api_server --port 24856 --model meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 52 --tensor_parallel_size 1 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests
./figure15_run.sh sys
```

You can use the following script to collect the data from the logs
```bash
#!/bin/bash
grep -r "Mean TTFT" logs/latency | sort -h
grep -r "Mean TPOT" logs/latency | sort -h
grep -r "Mean E2EL" logs/latency | sort -h
```