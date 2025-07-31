# SOSP'25 Jenga Artifact

## 1. Overview
This repository contains the artifact for the paper "Jenga: Effective Memory Management for Serving LLM with Heterogeneity" to be appeared in SOSP'25.

## 2. Environment Preparation
We have put all related environment, including both code and experiment scripts in a docker https://hub.docker.com/repository/docker/heheda12345/sosp-ae/general. Most experiments can be done with 1 H100 GPU, while some needs 1 L4 GPU, 8 H100 GPU, or 8 H200 GPU.

NOTE: for AE reviewers, please read the internal guide for launching that docker with our GPU resources.

Here is a step-by-step guide for setup the environment. You can skip this part if using our docker.




## 3. 


python3 -m vllm.entrypoints.openai.api_server xxxxx to launch a server. When you see `INFO:     Application startup complete.` in the log, the server starts and you can run the second command in the other terminal to run the experiments.
