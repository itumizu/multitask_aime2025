#!/bin/bash
# source .env
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
# unset CUDA_VISIBLE_DEVICES
echo "start: Nvidia CUDA MPS Control"