# Figure 15: Averaged Latency for the Llama Vision Model with changing request rates
Note that it needs several hours to reproduce this figure.

## 1. vLLM Baseline
Commands for setting up the environment.
```bash
source ~/venv/vllm-v0-baseline/bin/activate 
export HF_TOKEN=xxxxxxx # For AE reviewers: we give you an HF token in the internal guide.
cd Jenga-SOSP25-AE
```
Run the experiment.
```bash
# vLLM
python3 -m vllm.entrypoints.openai.api_server --port 24855 --model meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 52 --tensor_parallel_size 1 --disable-log-requests
./figure15_run.sh vllm
```

### 2. Max page, static partition, and Jenga
Commands for setting up the environment.
```bash
source ~/venv/vllm-v0-jenga/bin/
export HF_TOKEN=xxxxxxx # For AE reviewers: we give you an HF token in the internal guide.
cd Jenga-SOSP25-AE
```
Run the experiment.
```bash
# Max page
python3 -m vllm.entrypoints.openai.api_server --port 24857 --model meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 52 --tensor_parallel_size 1 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests --max-page-allocator
./figure15_run.sh max

# static partition
python3 -m vllm.entrypoints.openai.api_server --port 24858 --model meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 52 --tensor_parallel_size 1 --disable-v2-block-manager --use-per-layer-block-manager --enable-two-level-page --disable-log-requests --static-partition-allocator
./figure15_run.sh static

# Jenga
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