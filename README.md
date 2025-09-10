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

## Key Improvements

### 1. Enhanced Documentation
All modules now include comprehensive docstrings explaining:
- Module purpose
- Function parameters and return values
- Usage examples

### 2. Type Hints
Added type hints to improve code clarity and enable better IDE support:
```python
def process_dataset(
    dataset: List[Dict[str, Any]], 
    base_url: str, 
    model: str, 
    max_workers: int
) -> List[Dict[str, Any]]:
```

### 3. Better Error Handling
Improved error handling with more informative error messages:
```python
try:
    completion = client.chat.completions.create(...)
except Exception as e:
    print(f'Retry attempt {attempt+1} failed: {str(e)}')
    if attempt == max_retries:
        print("Max retries exceeded")
        return sample
```

### 4. Modular Design
Refactored code into logical functions with single responsibilities:
- Data loading and saving functions
- Data processing functions
- Model interaction functions
- Evaluation functions

### 5. Fixed Bugs
- Fixed string literal bug in `format_bmark.py`
- Corrected syntax issues in `obtain_think.py`

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

## Testing

All refactored code has been tested for:
- Syntax errors
- Module imports
- Basic functionality

For detailed information about the refactoring changes, see `REFACTORED_CHANGES.md`.