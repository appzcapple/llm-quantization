# Qwen2-VL-2B AWQ Quantization

This script quantizes the Qwen2-VL-2B-Instruct model to AWQ format for efficient inference.

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```bash
# Basic quantization with default settings
python quantize_qwen2vl_awq.py

# Custom output directory
python quantize_qwen2vl_awq.py --output_dir ./my_quantized_model

# 8-bit quantization instead of 4-bit
python quantize_qwen2vl_awq.py --w_bit 8

# Use custom calibration dataset
python quantize_qwen2vl_awq.py --calibration_dataset ./my_calib_data.json
```

## Parameters

- `--model_path`: HuggingFace model path (default: "Qwen/Qwen2-VL-2B-Instruct")
- `--output_dir`: Output directory (default: "./qwen2vl-2b-awq")
- `--w_bit`: Weight bit width, 4 or 8 (default: 4)
- `--q_group_size`: Quantization group size (default: 128)
- `--num_samples`: Number of calibration samples (default: 512)
- `--calibration_dataset`: Custom calibration dataset JSON file path
- `--device`: Device to use: auto/cuda/cpu (default: auto)

## Custom Calibration Dataset Format

```json
[
  {"text": "Sample text for calibration..."},
  {"text": "Another sample text..."}
]
```

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 20GB+ free disk space

## Loading the Quantized Model

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized("./qwen2vl-2b-awq")
tokenizer = AutoTokenizer.from_pretrained("./qwen2vl-2b-awq")
```