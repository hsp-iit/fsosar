#!/bin/bash

# Define model types to iterate over
models=("softmax" "eos" "gc" "discriminator")

# Base job script file name
job_script="train_safsar.sh"

# Loop over each model type and submit the job
for model in "${models[@]}"; do
    job_name="safsar_${model}"
    qsub -N "$job_name" -v OS_LOSS="$model" "$job_script"
done