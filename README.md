# Few-Shot Open-Set Action Recognition (FSOSAR)

<div align="center">
  <img src="methods.png" alt="FSOSAR Methods Overview" width="800"/>
  <p><i>Overview of implicit, garbage class, and discriminator methods for few-shot open-set action recognition</i></p>
</div>

---

## 📄 About

**This work has been submitted for review at the ICPR 2026 conference.**

This repository contains the implementation of few-shot learning methods adapted for open-set action recognition. The codebase supports various approaches to handle unknown classes during few-shot video classification tasks.

## 🎯 Supported Methods

The repository implements few-shot action recognition models:

- **SAFSAR** - Self-Attention Few-Shot Action Recognition
- **STRM** - Spatiotemporal Relational Matching

## 📊 Supported Datasets

- **SSv2** - Something-Something V2
- **HMDB51** - Human Motion Database
- **UCF101** - UCF Action Recognition Dataset
- **NTURGBD120** - NTU RGB+D 120 Action Recognition
- **Diving48** - Fine-grained Diving Dataset

## 🔬 Open-Set Loss Functions

Four different approaches for handling open-set scenarios:

- **softmax** - Standard softmax baseline (implicit method)
- **eos** - Entropic Open-Set loss
- **discriminator** - Binary discriminator approach
- **gc** - Garbage Class method

---

## 🛠️ Installation

### Prerequisites

- Conda or Miniconda
- CUDA-compatible GPU (recommended for training)

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd fsosar
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate fsosar
```

The environment includes:
- PyTorch with CUDA support
- Transformers (Hugging Face)
- CLIP (OpenAI)
- scikit-learn
- wandb (Weights & Biases)
- OpenCV, matplotlib, einops
- imageio

### License Information

This project is licensed under the **BSD 3-Clause License**. See [LICENSE](LICENSE) for details.

For information about dependencies and their licenses, see [LICENSE_DEPENDENCIES.md](LICENSE_DEPENDENCIES.md).

---

## 🚀 Usage

### Single Training Job

For training a single model configuration:

```bash
python train.py --model SAFSAR --data Diving48 --os_loss discriminator
```

**Arguments:**
- `--model`: Choose from `SAFSAR`, `STRM`
- `--data`: Choose from `SSv2`, `HMDB51`, `UCF101`, `NTURGBD120`, `Diving48`
- `--os_loss`: Choose from `softmax`, `eos`, `discriminator`, `gc`

### SLURM Cluster Training

#### Single Job Submission

For SLURM-based clusters, use the provided script in the `jobs/` directory:

```bash
cd jobs
sbatch train_single_slurm.sh
```

Edit the script to customize:
- Job name, partition, and resource allocation
- Model, dataset, and open-set loss parameters
- Time limits and GPU requirements

#### Batch Job Submission

For running multiple experiments in parallel:

```bash
cd jobs
./submit_batch_slurm.sh
```

**Customization options:**

```bash
# Default usage (uses configurations in the script)
./submit_batch_slurm.sh

# Custom models and datasets
./submit_batch_slurm.sh --models SAFSAR,STRM --datasets UCF101,HMDB51 --os-losses softmax,gc

# Custom resource allocation
./submit_batch_slurm.sh --partition gpuv --gpus 4 --cpus 16 --memory 32G --time 48:00:00

# View help
./submit_batch_slurm.sh --help
```

**Available options:**
- `--models MODEL1,MODEL2,...` - Comma-separated list of models
- `--datasets DATA1,DATA2,...` - Comma-separated list of datasets  
- `--os-losses LOSS1,LOSS2,...` - Comma-separated list of open-set losses
- `--partition PARTITION` - SLURM partition (default: gpu)
- `--time TIME` - Time limit (default: 24:00:00)
- `--cpus CPUS` - CPUs per task (default: 20)
- `--gpus GPUS` - Number of GPUs (default: 4)
- `--memory MEMORY` - Memory allocation (default: 64G)

The batch submission script will:
1. Validate all input parameters
2. Display all job combinations that will be submitted
3. Ask for confirmation before submission
4. Submit jobs to SLURM with proper resource allocation
5. Provide job IDs and monitoring commands

**Useful SLURM commands:**
```bash
squeue -u $USER                 # Check your job queue
squeue -j <job_id>              # Check specific job status
scancel <job_id>                # Cancel a job
scancel -u $USER                # Cancel all your jobs
scontrol show job <job_id>      # Show detailed job info
```

---

## 📁 Project Structure

```
fsosar/
├── LICENSE                     # BSD 3-Clause License
├── LICENSE_DEPENDENCIES.md     # Comprehensive license report for all dependencies
├── README.md                   # This file
├── environment.yaml            # Conda environment specification
├── methods.png                 # Methods diagram
├── configs/                    # Configuration files for models and datasets
│   ├── SAFSAR.json
│   ├── STRM.json
│   ├── SSv2.json
│   ├── HMDB51.json
│   ├── UCF101.json
│   ├── Diving48.json
│   ├── NTURGBD120.json
│   └── README.md
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── safsar.py
│   └── strm.py
├── jobs/                       # SLURM job scripts
│   ├── train_single_slurm.sh   # Single job submission
│   ├── train_batch_slurm.sh    # Batch worker script
│   ├── submit_batch_slurm.sh   # Batch submission manager
│   └── download_checkpoints.sh # Download pretrained checkpoints
├── splits/                     # Dataset split files
│   ├── diving/
│   ├── hmdb_ARN/
│   ├── ucf_ARN/
│   ├── ssv2_OTAM/
│   ├── kinetics_CMN/
│   └── nturgbd/
├── data/                       # Data preparation scripts
│   ├── prepare_diving48.py
│   ├── extract_images_from_videos.py
│   ├── create_train_test_diving_ntu.py
│   ├── get_classes_splits_classes_for_paper.py
│   ├── visualize_confusion_matrix.py
│   ├── visualize_confidence_histograms.py
│   └── visualize_features.py
├── videotransforms/            # Video augmentation utilities (custom module)
│   ├── video_transforms.py
│   ├── functional.py
│   ├── stack_transforms.py
│   └── ...
├── data_analysis/              # Analysis and visualization outputs
│   ├── confusion_matrices/
│   ├── histograms/
│   ├── saved_features/
│   └── confidence_scores/
├── checkpoints/                # Saved model checkpoints
│   ├── SAFSAR/
│   └── strm/
├── train.py                    # Main training script
├── videoloader.py              # Dataset loading utilities
├── utils.py                    # Helper functions
└── log_filter.py               # Log filtering utility
```

---

## ⚙️ Configuration

Model and dataset configurations are stored in the `configs/` directory as JSON files. Each configuration file contains:

- Model architecture parameters
- Training hyperparameters (learning rate, batch size, etc.)
- Dataset-specific settings
- Evaluation parameters
- Open-set loss specific configurations

To modify training behavior, edit the corresponding configuration files or override parameters in the training script.

---

## 📈 Monitoring and Logging

The repository supports Weights & Biases (wandb) for experiment tracking:

- Training/validation metrics
- Confusion matrices
- Confidence score histograms
- Feature visualizations
- OSCR curves and AUPR scores

Checkpoints are automatically saved to:
```
logs/<shot>_<model>_<os_loss>_<dataset>_<timestamp>/
```

---

## 📊 Analysis and Visualization

Several visualization scripts are provided in the `data/` directory:

- `data/visualize_confusion_matrix.py` - Generate confusion matrices
- `data/visualize_confidence_histograms.py` - Analyze prediction confidence
- `data/visualize_features.py` - t-SNE/UMAP feature space visualization

Results are saved to the `data_analysis/` directory.

---

## 🤝 Citation

If you use this code for your research, please cite our paper (citation will be added after acceptance).

---

## 📝 License

This project is licensed under the **BSD 3-Clause License**.  
Copyright (c) 2025, Istituto Italiano di Tecnologia

See [LICENSE](LICENSE) for the full license text.

All dependencies use permissive licenses compatible with BSD 3-Clause. For detailed information about third-party licenses, see [LICENSE_DEPENDENCIES.md](LICENSE_DEPENDENCIES.md).

---

## 🙏 Acknowledgments

This work builds upon several open-source implementations:
- CLIP by OpenAI
- Hugging Face Transformers
- PyTorch

---

## 📧 Contact

For questions or issues, please open an issue in the repository or contact the authors.

---

**Status:** Under review at ICPR 2026
