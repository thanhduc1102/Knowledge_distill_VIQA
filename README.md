# VLSP 2025 Financial Numerical Reasoning

This repository contains code for the VLSP 2025 Financial Numerical Reasoning task.

## Repository Structure

```
.
├── data/
│   ├── finqa/           # Original FinQA dataset
│   ├── process/         # Processed data
│   └── receive/         # Received data
├── src/                 # Source code
│   ├── assets/          # Template files
│   ├── serve/           # Serving scripts
│   └── *.py             # Python modules
└── README.md            # This file
```

### Core Modules

1. **`bmark.py`** - Benchmark evaluation script for financial reasoning models
2. **`format_bmark.py`** - Format data for benchmark evaluation
3. **`format_grpo.py`** - Format data for GRPO (Generalized Reward Policy Optimization) training
4. **`format_sft.py`** - Format data for SFT (Supervised Fine-Tuning) training
5. **`format_think.py`** - Format data for thinking process training
6. **`merge_finqa.py`** - Merge FinQA dataset splits
7. **`obtain_think.py`** - Generate thinking process data using LLMs
8. **`program_tokenizer.py`** - Tokenize program strings for evaluation
9. **`utils.py`** - Utility functions for file I/O operations

### Assets

1. **`assets/template.py`** - Template prompt for financial reasoning tasks
2. **`assets/think.py`** - Template prompt for thinking process generation

### Serving Scripts

The `serve/` directory contains scripts for serving models with vLLM:

- **`deepseek.sh`** - Serve DeepSeek model
- **`gemma.sh`** - Serve Gemma model
- **`glm.sh`** - Serve GLM model
- **`llm.sh`** - Serve Qwen model
- **`mistral.sh`** - Serve Mistral model
- **`nvidia.sh`** - Serve NVIDIA model
- **`serving.sh`** - Generic serving script
- **`test_api.py`** - Test API connectivity

## Dependencies

The code requires the following Python packages:
- `openai` - For API interactions
- `pandas` - For data processing
- `tqdm` - For progress bars
- `pyarrow` - For Parquet file support

Install with:
```bash
pip install openai pandas tqdm pyarrow
```

## Usage

### Benchmark Evaluation
```bash
python src/bmark.py \
  --ifp data/process/0802/bmark.json \
  --ofp_predictions data/process/0802/bmark/predictions.json \
  --ofp_results data/process/0802/bmark/results.json
```

### Data Formatting
```bash
python src/format_bmark.py
python src/format_grpo.py
python src/format_sft.py
python src/format_think.py
```

### Data Processing
```bash
python src/merge_finqa.py
python src/obtain_think.py
```

## Model Serving

Use the scripts in the `serve/` directory to serve models with vLLM:

```bash
# Serve a Qwen model
./src/serve/llm.sh
```