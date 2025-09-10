
#!/bin/bash

# Function to check if element exists in array
contains_element() {
    local element="$1"
    shift
    local array=("$@")
    for item in "${array[@]}"; do
        if [[ "$item" == "$element" ]]; then
            return 0
        fi
    done
    return 1
}

# Define valid models and datasets
valid_models=("STRM" "SAFSAR" "ActionCLIP" "MAML")
valid_datasets=("SSv2" "HMDB51" "UCF101" "NTURGBD120" "Diving48")
valid_os_losses=("softmax" "eos" "objectosphere" "discriminator" "gc")

# Define models to test (can be customized)
models=("STRM" "SAFSAR" "ActionCLIP" "MAML")

# Define datasets to test (can be customized)
datasets=("UCF101" "HMDB51")

# Define OS losses to test (can be customized)
os_losses=("softmax" "discriminator" "gc")

# Base job script file name
job_script="train_batch.sh"

# Validation function
validate_inputs() {
    echo "Validating inputs..."
    
    # Check models
    for model in "${models[@]}"; do
        if ! contains_element "$model" "${valid_models[@]}"; then
            echo "Error: Invalid model '$model'. Valid models are: ${valid_models[*]}"
            exit 1
        fi
    done
    
    # Check datasets
    for dataset in "${datasets[@]}"; do
        if ! contains_element "$dataset" "${valid_datasets[@]}"; then
            echo "Error: Invalid dataset '$dataset'. Valid datasets are: ${valid_datasets[*]}"
            exit 1
        fi
    done
    
    # Check OS losses
    for os_loss in "${os_losses[@]}"; do
        if ! contains_element "$os_loss" "${valid_os_losses[@]}"; then
            echo "Error: Invalid OS loss '$os_loss'. Valid OS losses are: ${valid_os_losses[*]}"
            exit 1
        fi
    done
    
    echo "All inputs are valid!"
}

# Function to display what will be run
show_combinations() {
    echo "Will run the following combinations:"
    echo "Models: ${models[*]}"
    echo "Datasets: ${datasets[*]}"
    echo "OS Losses: ${os_losses[*]}"
    echo "Total jobs: $((${#models[@]} * ${#datasets[@]} * ${#os_losses[@]}))"
    echo ""
    
    # Show detailed combinations
    local count=1
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for os_loss in "${os_losses[@]}"; do
                echo "$count. Model: $model, Dataset: $dataset, OS Loss: $os_loss"
                ((count++))
            done
        done
    done
    echo ""
}

# Validate inputs
validate_inputs

# Show what will be run
show_combinations

# Ask for confirmation
read -p "Do you want to submit these jobs? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Submit jobs
echo "Submitting jobs..."
job_count=0

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for os_loss in "${os_losses[@]}"; do
            job_name="${model}_${dataset}_${os_loss}"
            echo "Submitting job: $job_name"
            qsub -N "$job_name" -v MODEL="$model",DATA="$dataset",OS_LOSS="$os_loss" "$job_script"
            ((job_count++))
            
            # Add a small delay to avoid overwhelming the queue system
            sleep 1
        done
    done
done

echo "Successfully submitted $job_count jobs!"
echo "Use 'qstat' to check job status."
