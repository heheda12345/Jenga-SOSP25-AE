# SOSP'25 Jenga Artifact

## 1. Overview
This repository contains the artifact for the paper "Jenga: Effective Memory Management for Serving LLM with Heterogeneity" to be appeared in SOSP'25.
> Note: this repo is only for demo purpose. We have a simpified and more stable version in vLLM main branch and please  use that for deployment.

## 2. Environment Preparation
We have put all related environment, including both code and experiment scripts in a docker https://hub.docker.com/repository/docker/heheda12345/sosp-ae/general. Most experiments can be done with 1 H100 GPU, while some needs 1 L4 GPU, 8 H100 GPU, or 8 H200 GPU. After launching the docker, run `su ae` to switch to the user `ae`. Then you can see the environments (under `venv/`) and source code (`vllm-***`) in the home directory of the docker.

> NOTE for AE reviewers: please read the internal guide for launching that docker with our GPU resources.

[environment.md](environment.md) is a step-by-step guide for setup the environment. You can skip this part if using our docker.

## 3. Running the Experiments

### Basic Usage
Jenga is built on top of [VLLM](https://github.com/vllm-project/vllm), and you can use it as a normal VLLM server.
For each experiment, it contains 2 commands, one to launch the server, and the other to run the experiments. You should use 2 terminals to run them. Specifically,

1. In both terminals, activate the corresponding environment with the provided command in each experiment, and cd to the `Jenga-SOSP25-AE` directory.
2. In the first terminal, run the server with `python3 -m vllm.entrypoints.openai.api_server xxxxx`. When you see `INFO:     Application startup complete.` in the log, the server starts and you can run the second command in the other terminal to run the experiments.
3. In the second terminal, run the experiments with the provided command.
All scripts should be run under the `Jenga-SOSP25-AE` directory.

### Experiment Scripts
* [figure14.md](figure14.md) End-to-end throughput.
* [figure15.md](figure15.md) Averaged Latency for the Llama Vision Model with changing request rates.
* [figure19.md](figure19.md) Vision language model with chunked prefill.
* [figure20.md](figure20.md) Speculative decoding.
* [table2.md](table2.md) Max supported length of Llama 4 109B model.
> NOTE for AE reviewers: some experiments are pretty long. Please see the internal guide for the "quick" experiments that are suitable for your review.
