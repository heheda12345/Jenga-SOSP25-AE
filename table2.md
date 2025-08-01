# Table 2: Max supported length of Llama 4 109B model

Commands for setting up the environment.
```
source ~/venv/vllm-v1/bin/activate 
export HF_TOKEN=xxxxxxx # For AE reviewers: we give you an HF token in the internal guide.
cd Jenga-SOSP25-AE
```

You can try the following commands and see that it doesn't cause OOM. It needs 8 H100/H200 GPUs.
```bash
# vLLM H100
python3 -m vllm.entrypoints.openai.api_server --port 24867 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 13000000 --disable-log-requests --disable-hybrid-allocator 

# Jenga H100
python3 -m vllm.entrypoints.openai.api_server --port 24867 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 52000000 --disable-log-requests

# vLLM H200
python3 -m vllm.entrypoints.openai.api_server --port 24867 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 37000000 --disable-log-requests --disable-hybrid-allocator 

# Jenga H200
python3 -m vllm.entrypoints.openai.api_server --port 24867 --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tensor_parallel_size 8 --max-model-len 147000000 --disable-log-requests
```
