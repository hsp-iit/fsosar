#!/bin/bash

# Enhanced batch submission script for FSO-SAR experiments
# This script allows you to run multiple combinations of models, datasets, and open-set losses

# Function to print usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --models     Comma-separated list of models (default: all)"
    echo "  -d, --datasets   Comma-separated list of datasets (default: UCF101,HMDB51)"
    echo "  -o, --os-losses  Comma-separated list of open-set losses (default: softmax,discriminator)"
    echo "  -h, --help       Show this help message"
    echo "  --dry-run        Show what would be run without submitting jobs"
    echo "  --list-valid     List all valid options"
    echo ""
    echo "Examples:"
    echo "  $0 -m STRM,SAFSAR -d UCF101 -o softmax,discriminator"
    echo "  $0 --models ActionCLIP --datasets UCF101,HMDB51 --os-losses gc"
    echo "  $0 --dry-run"
    echo "  $0 --list-valid"
}

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

# Define valid options
valid_models=("STRM" "SAFSAR" "ActionCLIP" "MAML")
valid_datasets=("SSv2" "HMDB51" "UCF101" "NTURGBD120" "Diving48")
valid_os_losses=("softmax" "eos" "objectosphere" "discriminator" "gc")

# Function to list valid options
list_valid() {
    echo "Valid models: ${valid_models[*]}"
    echo "Valid datasets: ${valid_datasets[*]}"
    echo "Valid open-set losses: ${valid_os_losses[*]}"
}

# Default values
models=("STRM" "SAFSAR")
datasets=("UCF101" "HMDB51")
os_losses=("softmax" "discriminator")
dry_run=false

# Parse command line arguments
parse_array() {
    local input="$1"
    IFS=',' read -ra array <<< "$input"
    echo "${array[@]}"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            IFS=' ' read -ra models <<< "$(parse_array "$2")"
            shift 2
            ;;
        -d|--datasets)
            IFS=' ' read -ra datasets <<< "$(parse_array "$2")"
            shift 2
            ;;
        -o|--os-losses)
            IFS=' ' read -ra os_losses <<< "$(parse_array "$2")"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --list-valid)
            list_valid
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

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
    
    echo "✓ All inputs are valid!"
}

# Function to display what will be run
show_combinations() {
    echo ""
    echo "=== Experiment Configuration ==="
    echo "Models: ${models[*]}"
    echo "Datasets: ${datasets[*]}"
    echo "OS Losses: ${os_losses[*]}"
    echo "Total jobs: $((${#models[@]} * ${#datasets[@]} * ${#os_losses[@]}))"
    echo ""
    
    # Show detailed combinations
    echo "=== Job Details ==="
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

# Check if job script exists
job_script="train_batch.sh"
if [[ ! -f "$job_script" ]]; then
    echo "Error: Job script '$job_script' not found!"
    exit 1
fi

# Validate inputs
validate_inputs

# Show what will be run
show_combinations

# If dry run, exit here
if [[ "$dry_run" == true ]]; then
    echo "Dry run complete. No jobs were submitted."
    exit 0
fi

# Ask for confirmation
read -p "Do you want to submit these jobs? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Submit jobs
echo ""
echo "=== Submitting Jobs ==="
job_count=0
submitted_jobs=()

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for os_loss in "${os_losses[@]}"; do
            job_name="${model}_${dataset}_${os_loss}"
            echo "Submitting job: $job_name"
            
            # Submit job and capture job ID
            job_id=$(qsub -N "$job_name" -v MODEL="$model",DATA="$dataset",OS_LOSS="$os_loss" "$job_script")
            
            if [[ $? -eq 0 ]]; then
                submitted_jobs+=("$job_id ($job_name)")
                ((job_count++))
            else
                echo "Failed to submit job: $job_name"
            fi
            
            # Add a small delay to avoid overwhelming the queue system
            sleep 1
        done
    done
done

echo ""
echo "=== Summary ==="
echo "Successfully submitted $job_count jobs!"
echo ""
echo "Submitted job IDs:"
for job in "${submitted_jobs[@]}"; do
    echo "  $job"
done
echo ""
echo "Use 'qstat' to check job status."
echo "Use 'qstat -u \$USER' to see only your jobs."
