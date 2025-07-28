#!/bin/bash

set -x
mkdir -p logs/latency

for request_rate in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0; do
    if [ $1 == "sys" ]; then
        python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24856/v1 --endpoint /chat/completions --percentile-metrics ttft,tpot,itl,e2el --model  meta-llama/Llama-3.2-11B-Vision-Instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts 100 --seed 55555 --hf-output-len 200 --request-rate $request_rate --ignore-eos 2>&1 | tee logs/latency/mllama.sys.$request_rate.log
    elif [ $1 == "vllm" ]; then
        python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24855/v1 --endpoint /chat/completions --percentile-metrics ttft,tpot,itl,e2el --model  meta-llama/Llama-3.2-11B-Vision-Instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts 100 --seed 55555 --hf-output-len 200 --request-rate $request_rate --ignore-eos 2>&1 | tee logs/latency/mllama.vllm.$request_rate.log
    elif [ $1 == "max" ]; then
        python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24857/v1 --endpoint /chat/completions --percentile-metrics ttft,tpot,itl,e2el --model  meta-llama/Llama-3.2-11B-Vision-Instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts 100 --seed 55555 --hf-output-len 200 --request-rate $request_rate --ignore-eos 2>&1 | tee logs/latency/mllama.max.$request_rate.log
    elif [ $1 == "static" ]; then
        python3 benchmark_serving.py --backend openai-chat --base-url http://localhost:24858/v1 --endpoint /chat/completions --percentile-metrics ttft,tpot,itl,e2el --model  meta-llama/Llama-3.2-11B-Vision-Instruct --dataset-path MMMU/MMMU_Pro --dataset-name hf --hf-subset "vision" --hf-split test --num_prompts 100 --seed 55555 --hf-output-len 200 --request-rate $request_rate --ignore-eos 2>&1 | tee logs/latency/mllama.static.$request_rate.log
    fi
done