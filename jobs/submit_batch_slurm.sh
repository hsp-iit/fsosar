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

# Reservation configuration (set to empty string to disable)
reservation_name="sberti_14"  # Set to "" to disable reservation usage

# Job batching configuration
jobs_with_reservation=16    # Number of jobs to submit with reservation
jobs_without_reservation=6  # Number of jobs to submit without reservation

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
time_limit="12:00:00"
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
    if [[ -n "$reservation_name" ]]; then
        echo "Reservation: $reservation_name (pattern: $jobs_with_reservation with / $jobs_without_reservation without)"
    else
        echo "Reservation: None"
    fi
    echo ""
    echo "Job combinations:"
    echo "Models: ${models[*]}"
    echo "Datasets: ${datasets[*]}"
    echo "OS Losses: ${os_losses[*]}"
    echo "Total jobs: $((${#models[@]} * ${#datasets[@]} * ${#os_losses[@]}))"
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

# Submit jobs automatically (no user confirmation needed)
echo "Submitting jobs to SLURM..."
job_count=0
job_ids=()

# Create logs directory if it doesn't exist
logs_dir="/fastwork/sberti/fsosar/slurm_logs"
mkdir -p "$logs_dir"

# Create array of all job combinations
job_combinations=()
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for os_loss in "${os_losses[@]}"; do
            job_combinations+=("$model:$dataset:$os_loss")
        done
    done
done

echo "Total job combinations: ${#job_combinations[@]}"
if [[ -n "$reservation_name" ]]; then
    echo "Using reservation pattern: $jobs_with_reservation jobs with '$reservation_name', then $jobs_without_reservation jobs without"
else
    echo "No reservation will be used"
fi
echo "Starting job submission..."
echo ""

# Submit jobs with alternating reservation pattern
reservation_counter=0
for job_combo in "${job_combinations[@]}"; do
    IFS=':' read -r model dataset os_loss <<< "$job_combo"
    job_name="${model}_${dataset}_${os_loss}"
    
    # Determine if we should use reservation for this job
    use_reservation=false
    if [[ -n "$reservation_name" ]]; then
        cycle_position=$((reservation_counter % (jobs_with_reservation + jobs_without_reservation)))
        if [[ $cycle_position -lt $jobs_with_reservation ]]; then
            use_reservation=true
        fi
    fi
    
    echo "Submitting job: $job_name"
    if [[ "$use_reservation" == true ]]; then
        echo "  → With reservation: $reservation_name"
        # Submit job with reservation
        job_id=$(sbatch \
            --job-name="$job_name" \
            --partition="$partition" \
            --reservation="$reservation_name" \
            --time="$time_limit" \
            --cpus-per-task="$cpus_per_task" \
            --gres="gpu:$gpus" \
            --mem="$memory" \
            --output="$logs_dir/${job_name}_%j.out" \
            --error="$logs_dir/${job_name}_%j.err" \
            --export="MODEL=$model,DATA=$dataset,OS_LOSS=$os_loss" \
            "$(dirname "$0")/$job_script" | awk '{print $4}')
    else
        echo "  → No reservation"
        # Submit job without reservation
        job_id=$(sbatch \
            --job-name="$job_name" \
            --partition="$partition" \
            --time="$time_limit" \
            --cpus-per-task="$cpus_per_task" \
            --gres="gpu:$gpus" \
            --mem="$memory" \
            --output="$logs_dir/${job_name}_%j.out" \
            --error="$logs_dir/${job_name}_%j.err" \
            --export="MODEL=$model,DATA=$dataset,OS_LOSS=$os_loss" \
            "$(dirname "$0")/$job_script" | awk '{print $4}')
    fi
    
    if [[ -n "$job_id" ]]; then
        job_ids+=("$job_id")
        echo "  → Job ID: $job_id"
    else
        echo "  → FAILED"
    fi
    
    ((job_count++))
    ((reservation_counter++))
    sleep 0.3  # Brief pause
done

echo ""
echo "==============================================="
echo "Job Submission Summary"
echo "==============================================="
echo "Successfully submitted $job_count jobs!"
echo "Job IDs: ${job_ids[*]}"
echo ""
echo "Log files location: $logs_dir"
echo ""
echo "Useful SLURM commands:"
echo "  squeue -u \$USER          # Check your job queue"
echo "  squeue -j <job_id>        # Check specific job status"
echo "  scancel <job_id>          # Cancel a specific job"
echo "  scancel -u \$USER         # Cancel all your jobs"
echo "  scontrol show job <job_id> # Show detailed job info"
echo ""
echo "Job output files: $logs_dir/<job_name>_<job_id>.out"
echo "Job error files: $logs_dir/<job_name>_<job_id>.err"
echo "==============================================="
