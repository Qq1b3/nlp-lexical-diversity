# Milestone 2: Baseline Methods for Text Classification

This directory contains the implementation of baseline methods for distinguishing between pre-ChatGPT and post-ChatGPT news articles.

## Overview

We implement and compare multiple classification approaches:
- **Machine Learning methods**: Logistic Regression, Naive Bayes, SVM, Neural Networks
- **Rule-based methods**: Date threshold, text length, lexical diversity

## Scripts

### `ml_baselines.py`
Main ML baseline implementation with the following classifiers:
- Logistic Regression with TF-IDF features
- Naive Bayes with TF-IDF features
- SVM (Support Vector Machine) with TF-IDF features
- PyTorch MLP with TF-IDF features (GPU-accelerated)
- All above models with additional lexical diversity features (TTR, HD-D, MTLD, VocD)

**Usage:**
```bash
# Full dataset (recommended for final results)
python task2/ml_baselines.py --no-test --epochs 100

# Test mode (5% of data for quick testing)
python task2/ml_baselines.py --test --test-percent 5 --epochs 10
```

**Arguments:**
- `--test` / `--no-test`: Enable/disable test mode (default: disabled)
- `--test-percent`: Percentage of data to use in test mode (default: 5)
- `--epochs`: Number of training epochs for neural network models (default: 100)

### `rule_baselines.py`
Rule-based classification methods:
- **Date Threshold**: Classifies based on publication date (before/after Nov 2022)
- **Text Length**: Uses character count threshold
- **Lexical Diversity**: Uses Type-Token Ratio (TTR) threshold

**Usage:**
```bash
python task2/rule_baselines.py --no-test
```

### `compare_methods.py`
Compares all ML and rule-based methods side-by-side:
- Loads predictions from all methods
- Computes pairwise agreement/disagreement statistics
- Generates summary comparison table

**Usage:**
```bash
python task2/compare_methods.py
```

## Results

Results are stored in the `results/` directory (download from cloud - not tracked in Git).

**Download results:** [TUCloud Link] (ask team for access)

### Result Files:
- `ml_baseline_results.json` - Quantitative metrics for all ML models
- `rule_baseline_results.json` - Quantitative metrics for rule-based methods
- `all_methods_comparison.json` - Combined comparison of all methods
- `predictions_*.json` - Per-sample predictions for each method
- `ml_baseline_misclassifications.json` - Analysis of misclassified examples

## Requirements

Additional dependencies for Milestone 2 (beyond base requirements):
```
scikit-learn>=1.0.0
torch>=2.0.0
wandb>=0.15.0
pynvml>=11.0.0
psutil>=5.8.0
lexicalrichness>=0.1.0
```

### Optional: GPU Acceleration
For GPU-accelerated training, install:
- PyTorch with CUDA support
- cuML (RAPIDS) for GPU-accelerated scikit-learn models

```bash
# PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# cuML (requires conda)
conda install -c rapidsai -c conda-forge -c nvidia cuml
```

## Execution Environment

**Note:** All experiments were run on a Linux server (Ubuntu 24.04) with dual NVIDIA RTX 4090 GPUs due to:
- Better GPU/CPU performance for training on 1.2M articles
- Windows compatibility issues with cuML (RAPIDS GPU library)
- cuML requires conda and has limited Windows support

### Server Specifications:
- **OS:** Ubuntu 24.04 LTS
- **GPU:** 2x NVIDIA GeForce RTX 4090 (24GB VRAM each)
- **CPU:** 32 cores
- **CUDA:** 13.0
- **Environment:** Miniconda with Python 3.10

### Running on Linux Server:

```bash
# Create conda environment (required for cuML)
conda create -n nlp-project python=3.10
conda activate nlp-project

# Install cuML for GPU-accelerated ML
conda install -c rapidsai -c conda-forge -c nvidia cuml

# Install other requirements
pip install -r requirements.txt

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Experiment Tracking

We use Weights & Biases (wandb) for experiment tracking, logging:
- Training metrics (accuracy, F1, precision, recall)
- Confusion matrices
- GPU utilization, memory, and temperature
- CPU and RAM usage
- Terminal output

### View Experiments:
**Dashboard:** https://wandb.ai/jojino-tu-wien/nlp-lexical-diversity-milestone2

**Sharing with teammates:**
- Teammates can create a free wandb account and request access via the link above
- Alternatively, screenshots of key metrics are included in the results folder

### Setup wandb locally:
```bash
pip install wandb
wandb login
```

## Results Summary

| Method | Accuracy | F1 (weighted) |
|--------|----------|---------------|
| Logistic Regression + Lexical | **62.09%** | 56.89% |
| Logistic Regression TF-IDF | 62.06% | 56.83% |
| Naive Bayes + Lexical | 61.23% | 56.27% |
| Naive Bayes TF-IDF | 61.17% | 56.19% |
| PyTorch MLP (CUDA) | 60.17% | **59.73%** |
| SVM | 60.12% | 48.96% |
| Rule: Lexical Diversity | 57.40% | 41.91% |
| Rule: Text Length | 42.68% | 27.40% |

**Key Finding:** Pre-ChatGPT and post-ChatGPT news articles are linguistically very similar. The best ML model achieves only ~62% accuracy, suggesting minimal stylistic differences between the two periods.

## Dataset

- **Total articles:** 1,214,965
- **Languages:** English, Czech
- **Train/Test split:** 80/20 (971,972 / 242,993)
- **Class distribution:**
  - Pre-ChatGPT: 517,666 (42.6%)
  - Post-ChatGPT: 697,299 (57.4%)

