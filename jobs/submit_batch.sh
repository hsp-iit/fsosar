
#!/bin/bash
# Define model types to iterate over
models=("softmax" "eos" "gc" "discriminator")

# Define data type to iterate over
data="UCF101"

# Base job script file name
job_script="train_batch.sh"

# Loop over each model type and submit the job
for model in "${models[@]}"; do
    job_name="safsar_${data}_${model}"
    qsub -N "$job_name" -v OS_LOSS="$model",DATA="$data" "$job_script"
done
