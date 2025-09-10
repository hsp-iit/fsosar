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
models=("ActionCLIP" "MAML")

# Define datasets to test (can be customized)
datasets=("UCF101" "HMDB51" "Diving48" "NTURGBD120" "SSv2")

# Define OS losses to test (can be customized)
os_losses=("softmax" "eos" "discriminator" "gc")

# Base job script file name (SLURM version)
job_script="train_batch_slurm.sh"

# SLURM job configuration (can be customized)
partition="gpuv"
time_limit="6:00:00"
cpus_per_task=10
gpus=1
memory="32G"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            IFS=',' read -ra models <<< "$2"
            shift 2
            ;;
        --datasets)
            IFS=',' read -ra datasets <<< "$2"
            shift 2
            ;;
        --os-losses)
            IFS=',' read -ra os_losses <<< "$2"
            shift 2
            ;;
        --partition)
            partition="$2"
            shift 2
            ;;
        --time)
            time_limit="$2"
            shift 2
            ;;
        --cpus)
            cpus_per_task="$2"
            shift 2
            ;;
        --gpus)
            gpus="$2"
            shift 2
            ;;
        --memory)
            memory="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models MODEL1,MODEL2,...    Comma-separated list of models to test"
            echo "  --datasets DATA1,DATA2,...    Comma-separated list of datasets to test"
            echo "  --os-losses LOSS1,LOSS2,...   Comma-separated list of OS losses to test"
            echo "  --partition PARTITION          SLURM partition to use (default: gpu)"
            echo "  --time TIME                    Time limit (default: 24:00:00)"
            echo "  --cpus CPUS                    CPUs per task (default: 20)"
            echo "  --gpus GPUS                    Number of GPUs (default: 4)"
            echo "  --memory MEMORY                Memory per job (default: 64G)"
            echo "  --help                         Show this help message"
            echo ""
            echo "Valid models: ${valid_models[*]}"
            echo "Valid datasets: ${valid_datasets[*]}"
            echo "Valid OS losses: ${valid_os_losses[*]}"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --models STRM,SAFSAR --datasets UCF101 --os-losses softmax,discriminator"
            echo "  $0 --models ActionCLIP --datasets UCF101,HMDB51 --os-losses gc"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Use --help for usage information"
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
    
    echo "All inputs are valid!"
}

# Function to display what will be run
show_combinations() {
    echo "==============================================="
    echo "SLURM Job Submission Configuration"
    echo "==============================================="
    echo "Partition: $partition"
    echo "Time limit: $time_limit"
    echo "CPUs per task: $cpus_per_task"
    echo "GPUs per job: $gpus"
    echo "Memory per job: $memory"
    echo ""
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
    echo "==============================================="
}

# Function to check if SLURM is available
check_slurm() {
    if ! command -v sbatch &> /dev/null; then
        echo "Error: SLURM (sbatch) is not available on this system."
        echo "Make sure SLURM is properly installed and configured."
        exit 1
    fi
}

# Check if job script exists
check_job_script() {
    local script_path="$(dirname "$0")/$job_script"
    if [[ ! -f "$script_path" ]]; then
        echo "Error: Job script '$script_path' not found."
        echo "Make sure $job_script exists in the same directory as this script."
        exit 1
    fi
}

# Validate SLURM environment
check_slurm
check_job_script

# Validate inputs
validate_inputs

# Show what will be run
show_combinations

# Ask for confirmation
read -p "Do you want to submit these jobs to SLURM? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Submit jobs
echo "Submitting jobs to SLURM..."
job_count=0
job_ids=()

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for os_loss in "${os_losses[@]}"; do
            job_name="${model}_${dataset}_${os_loss}"
            echo "Submitting job: $job_name"
            
            # Submit job with custom configuration
            job_id=$(sbatch \
                --job-name="$job_name" \
                --partition="$partition" \
                --time="$time_limit" \
                --cpus-per-task="$cpus_per_task" \
                --gres="gpu:$gpus" \
                --mem="$memory" \
                --export="MODEL=$model,DATA=$dataset,OS_LOSS=$os_loss" \
                "$(dirname "$0")/$job_script" | awk '{print $4}')
            
            if [[ -n "$job_id" ]]; then
                job_ids+=("$job_id")
                echo "  → Job ID: $job_id"
            else
                echo "  → Failed to submit job"
            fi
            
            ((job_count++))
            
            # Add a small delay to avoid overwhelming the queue system
            sleep 0.5
        done
    done
done

echo ""
echo "==============================================="
echo "Job Submission Summary"
echo "==============================================="
echo "Successfully submitted $job_count jobs!"
echo "Job IDs: ${job_ids[*]}"
echo ""
echo "Useful SLURM commands:"
echo "  squeue -u \$USER          # Check your job queue"
echo "  squeue -j <job_id>        # Check specific job status"
echo "  scancel <job_id>          # Cancel a specific job"
echo "  scancel -u \$USER         # Cancel all your jobs"
echo "  scontrol show job <job_id> # Show detailed job info"
echo ""
echo "Job output files will be saved as: <job_name>_<job_id>.out"
echo "Job error files will be saved as: <job_name>_<job_id>.err"
echo "==============================================="
