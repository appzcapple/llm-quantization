#!/usr/bin/env python3
"""
Quantize Qwen2-VL-2B-Instruct model to AWQ format using AutoAWQ library.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoProcessor
from awq import AutoAWQForCausalLM
from datasets import load_dataset
import json


def load_calibration_dataset(dataset_path=None, num_samples=512):
    """Load calibration dataset for AWQ quantization."""
    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading custom calibration dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [item['text'] for item in data[:num_samples]]
    else:
        print("Loading default calibration dataset (wikitext)")
        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            return [item['text'] for item in dataset if len(item['text']) > 50][:num_samples]
        except Exception as e:
            print(f"Error loading wikitext dataset: {e}")
            print("Using simple text samples as fallback")
            return [
                "The quick brown fox jumps over the lazy dog.",
                "Python is a high-level programming language.",
                "Machine learning is a subset of artificial intelligence.",
                "Computer vision enables machines to interpret visual information.",
            ] * (num_samples // 4)


def quantize_qwen2vl_awq(
    model_path="Qwen/Qwen2-VL-2B-Instruct",
    output_dir="./qwen2vl-2b-awq",
    w_bit=4,
    q_group_size=128,
    num_samples=512,
    calibration_dataset=None,
    device="auto"
):
    """
    Quantize Qwen2-VL model to AWQ format.
    
    Args:
        model_path: HuggingFace model path or local path
        output_dir: Directory to save quantized model
        w_bit: Weight bit width (4 or 8)
        q_group_size: Group size for quantization
        num_samples: Number of calibration samples
        calibration_dataset: Path to custom calibration dataset JSON file
        device: Device to use ("auto", "cuda", "cpu")
    """
    
    print(f"Starting AWQ quantization for {model_path}")
    print(f"Configuration: w_bit={w_bit}, q_group_size={q_group_size}")
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer and processor
    print("Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Load model for AWQ quantization
    print("Loading model for quantization...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load calibration dataset
    print("Preparing calibration dataset...")
    calib_data = load_calibration_dataset(calibration_dataset, num_samples)
    print(f"Loaded {len(calib_data)} calibration samples")
    
    # Prepare quantization config
    quant_config = {
        "zero_point": True,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": "GEMM"
    }
    
    print("Starting quantization process...")
    
    # Quantize the model
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,
        split="train",
        text_column="text" if calibration_dataset else None
    )
    
    # Save quantized model
    print(f"Saving quantized model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # Save quantization config
    config_path = os.path.join(output_dir, "quantization_config.json")
    with open(config_path, 'w') as f:
        json.dump(quant_config, f, indent=2)
    
    print(f"✅ Quantization completed! Model saved to: {output_dir}")
    
    # Print model size comparison
    try:
        original_size = sum(p.numel() * p.element_size() for p in model.model.parameters()) / (1024**3)
        print(f"Estimated size reduction: ~{100 * (1 - w_bit/16):.1f}%")
    except:
        print("Size estimation unavailable")


def main():
    parser = argparse.ArgumentParser(description="Quantize Qwen2-VL-2B model to AWQ format")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model path or local model directory"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./qwen2vl-2b-awq",
        help="Output directory for quantized model"
    )
    
    parser.add_argument(
        "--w_bit", 
        type=int, 
        choices=[4, 8], 
        default=4,
        help="Weight quantization bit width"
    )
    
    parser.add_argument(
        "--q_group_size", 
        type=int, 
        default=128,
        help="Group size for quantization"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=512,
        help="Number of calibration samples"
    )
    
    parser.add_argument(
        "--calibration_dataset", 
        type=str,
        help="Path to custom calibration dataset (JSON format)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["auto", "cuda", "cpu"], 
        default="auto",
        help="Device to use for quantization"
    )
    
    args = parser.parse_args()
    
    # Check if CUDA is available when requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Using CPU instead.")
        args.device = "cpu"
    
    try:
        quantize_qwen2vl_awq(
            model_path=args.model_path,
            output_dir=args.output_dir,
            w_bit=args.w_bit,
            q_group_size=args.q_group_size,
            num_samples=args.num_samples,
            calibration_dataset=args.calibration_dataset,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Error during quantization: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())