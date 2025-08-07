# Environment Setup Instructions

## Installation Completed ✅

I have successfully installed:
1. **Miniconda** at `~/miniconda3/`
2. **Python environment** named `qwen2vl-awq` with Python 3.10
3. **Required packages**: PyTorch, transformers, datasets, accelerate, autoawq, and all dependencies

## How to Use

### 1. Activate the environment
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen2vl-awq
```

### 2. Verify installation
```bash
python -c "import torch; import transformers; import autoawq; print('All packages loaded successfully!')"
```

### 3. Run the quantization script
```bash
# Basic usage (will quantize Qwen2-VL-2B to 4-bit AWQ)
python quantize_qwen2vl_awq.py

# Custom settings
python quantize_qwen2vl_awq.py --output_dir ./my_quantized_model --w_bit 8
```

## Environment Management

- **Activate environment**: `conda activate qwen2vl-awq`
- **Deactivate environment**: `conda deactivate`
- **List environments**: `conda env list`
- **Delete environment** (if needed): `conda env remove -n qwen2vl-awq`

## Notes

- The environment is ready to use immediately
- AutoAWQ was installed without triton dependency (not needed on macOS)
- All required packages are properly installed and compatible