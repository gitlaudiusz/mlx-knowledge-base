# 🔄 Model Conversion Guide for MLX

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Basic Conversion](#basic-conversion)
3. [Quantization Options](#quantization-options)
4. [Advanced Scenarios](#advanced-scenarios)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

### Environment Setup
```bash
# Using UV (recommended)
uv venv && source .venv/bin/activate
uv add mlx mlx-lm huggingface_hub

# Traditional pip
pip install mlx mlx-lm huggingface_hub
```

### HuggingFace Login
```bash
huggingface-cli login
# Enter your token with write access if uploading
```

## Basic Conversion

### 1. Download Model from HuggingFace
```bash
# Download entire model
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/llama-2-7b

# Download specific files
huggingface-cli download meta-llama/Llama-2-7b-hf \
  --include "*.safetensors" "*.json" \
  --local-dir ./models/llama-2-7b
```

### 2. Convert to MLX Format
```bash
# Basic FP16 conversion
mlx_lm.convert \
  --hf-path ./models/llama-2-7b \
  --mlx-path ./models/llama-2-7b-mlx

# With 4-bit quantization
mlx_lm.convert \
  --hf-path ./models/llama-2-7b \
  --mlx-path ./models/llama-2-7b-mlx-4bit \
  --quantize \
  --q-bits 4
```

### 3. Test Converted Model
```bash
mlx_lm.generate \
  --model ./models/llama-2-7b-mlx-4bit \
  --prompt "Hello world" \
  --max-tokens 50
```

## Quantization Options

### Quantization Bits
```bash
# 8-bit (better quality, larger size)
--quantize --q-bits 8

# 4-bit (balanced quality/size)
--quantize --q-bits 4

# 2-bit (experimental, maximum compression)
--quantize --q-bits 2
```

### Group Size
```bash
# Smaller groups = better quality, slightly larger
--q-group-size 32

# Default
--q-group-size 64

# Larger groups = more compression
--q-group-size 128
```

### Mixed Precision (Python API)
```python
from mlx_lm.convert import convert

def custom_quantization(name, layer):
    # Keep embeddings and output in higher precision
    if "embed" in name or "lm_head" in name:
        return {"bits": 8, "group_size": 64}
    # Everything else in 4-bit
    return {"bits": 4, "group_size": 64}

convert(
    hf_path="model/path",
    mlx_path="output/path",
    quantize=True,
    quant_predicate=custom_quantization
)
```

## Advanced Scenarios

### Large Models (30B+)
```bash
# For models that don't fit in RAM during conversion
# Use lazy loading (automatic in recent versions)
mlx_lm.convert \
  --hf-path ./models/llama-30b \
  --mlx-path ./models/llama-30b-mlx-4bit \
  --quantize \
  --q-bits 4

# Monitor memory usage
watch -n 1 "ps aux | grep python | grep -v grep"
```

### Sharded Models
```python
# Some models come in multiple safetensors files
# MLX handles this automatically, but ensure all shards are present
ls ./models/large-model/
# Should show: model-00001-of-00005.safetensors, etc.
```

### Upload to HuggingFace
```bash
# Convert and upload in one step
mlx_lm.convert \
  --hf-path original/model \
  --mlx-path temporary/path \
  --quantize \
  --upload-repo your-username/model-mlx-4bit
```

### Batch Conversion Script
```bash
#!/bin/bash
# convert_batch.sh

models=("model1" "model2" "model3")
quantizations=(4 8)

for model in "${models[@]}"; do
  for q_bits in "${quantizations[@]}"; do
    echo "Converting $model with $q_bits-bit quantization..."
    mlx_lm.convert \
      --hf-path "./models/$model" \
      --mlx-path "./models/${model}-mlx-${q_bits}bit" \
      --quantize \
      --q-bits $q_bits
  done
done
```

## Special Cases

### Gemma Models
```bash
# Gemma models need special vocabulary handling
mlx_lm.convert \
  --hf-path google/gemma-7b \
  --mlx-path ./gemma-7b-mlx \
  --quantize
```

### Mistral/Mixtral
```bash
# Standard conversion works well
mlx_lm.convert \
  --hf-path mistralai/Mistral-7B-v0.1 \
  --mlx-path ./mistral-7b-mlx \
  --quantize
```

### Custom Models
If your model isn't supported, you may need to:
1. Check if architecture is implemented in MLX
2. File an issue or PR to add support
3. Use similar architecture as base

## Memory Requirements

### Rough Estimates for Conversion
| Model Size | FP16 RAM | 4-bit RAM | Conversion RAM |
|------------|----------|-----------|----------------|
| 7B         | 14GB     | 4GB       | ~20GB          |
| 13B        | 26GB     | 7GB       | ~35GB          |
| 30B        | 60GB     | 15GB      | ~80GB          |
| 70B        | 140GB    | 35GB      | ~180GB         |

### Tips for Limited RAM
1. Close other applications
2. Use 4-bit quantization
3. Convert on a machine with more RAM
4. Use pre-converted models from mlx-community

## Troubleshooting

### "No safetensors found"
```bash
# Ensure model has safetensors files
ls ./models/your-model/*.safetensors

# If only .bin files exist, model needs conversion to safetensors first
```

### Out of Memory
```bash
# Reduce memory pressure
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use system monitor
htop  # or Activity Monitor on macOS
```

### Slow Conversion
- Normal for large models (30B+ can take 30-60 minutes)
- Ensure you're using SSD, not HDD
- Don't run on battery power (reduced performance)

### Model Doesn't Generate Properly
```python
# Verify model structure
from mlx_lm import load
model, tokenizer = load("path/to/converted")
print(model)  # Check architecture
print(tokenizer.vocab_size)  # Verify vocabulary
```

## Best Practices

1. **Always Test After Conversion**
   - Generate sample text
   - Compare with original model if possible

2. **Document Your Conversions**
   - Note quantization settings
   - Record performance metrics
   - Save conversion commands

3. **Share with Community**
   - Upload successful conversions to HuggingFace
   - Use clear naming: `model-mlx-4bit`
   - Include model card with details

4. **Version Control**
   - Tag your conversions
   - Keep track of MLX version used
   - Note any special modifications

---

*Remember: Conversion is a one-time process. Once done, the MLX model loads much faster than the original!*