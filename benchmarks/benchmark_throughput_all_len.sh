#!/bin/bash
for input_len in 40000 50000 60000 70000 80000 90000 1000000 110000 120000; do
    python3 benchmark_throughput.py --model 'mistralai/Ministral-8B-Instruct-2410' --input-len 120000 --output-len 1024 --num-prompts 60 2>&1 | tee logs/benchmark.$input_len.log
done