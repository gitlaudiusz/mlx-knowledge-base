# 🚀 MLX Commands Cheatsheet

## Environment Setup
```bash
# UV (faster than pip)
uv venv && source .venv/bin/activate
uv add mlx mlx-lm mlx-vlm

# Traditional pip
pip install mlx mlx-lm mlx-vlm
```

## Model Operations

### Download from HuggingFace
```bash
huggingface-cli download model/name --local-dir ./models/model-name
```

### Convert to MLX
```bash
# Basic conversion (FP16)
mlx_lm.convert --hf-path model/path --mlx-path output/path

# With quantization
mlx_lm.convert --hf-path model/path --mlx-path output/path --quantize --q-bits 4 --q-group-size 64

# Upload to HF
mlx_lm.convert --hf-path input --mlx-path output --upload-repo username/model-name
```

### Generate Text
```bash
# Basic generation
mlx_lm.generate --model model/path --prompt "Your prompt"

# With parameters
mlx_lm.generate --model model/path \
  --prompt "Your prompt" \
  --max-tokens 512 \
  --temp 0.7 \
  --top-p 0.9

# Chat mode
mlx_lm.chat --model model/path
```

## Fine-tuning

### LoRA Training
```bash
# Basic LoRA
mlx_lm.lora --model base/model \
  --train \
  --data data/folder \
  --iters 1000 \
  --batch-size 2

# With custom parameters
mlx_lm.lora --model base/model \
  --train \
  --data data/folder \
  --iters 1000 \
  --batch-size 1 \
  --lora-layers 8 \
  --learning-rate 2e-4 \
  --warmup-steps 100
```

### Fuse Adapter
```bash
mlx_lm.fuse --model base/model \
  --adapter-path adapters/path \
  --save-path fused-model
```

## Memory Guidelines

### 16GB RAM
```bash
# Use 4-bit, batch=1
mlx_lm.lora --model 7b-model --batch-size 1 --lora-layers 4
```

### 32GB RAM
```bash
# 8-bit or 4-bit, batch=2-4
mlx_lm.lora --model 13b-model --batch-size 2 --lora-layers 8
```

### 64GB+ RAM
```bash
# Can handle larger models
mlx_lm.lora --model 30b-model --batch-size 4 --lora-layers 16
```

## Python API Examples

### Basic Generation
```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/model-4bit")
response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
print(response)
```

### With Caching
```python
from mlx_lm.models.cache import make_prompt_cache

cache = make_prompt_cache(model)
response = generate(model, tokenizer, prompt="Hello", prompt_cache=cache)
```

### Custom Quantization
```python
from mlx_lm.convert import convert

def mixed_quant(name, layer):
    if "lm_head" in name:
        return {"bits": 8, "group_size": 64}
    return {"bits": 4, "group_size": 64}

convert(hf_path="model", mlx_path="output", 
        quantize=True, quant_predicate=mixed_quant)
```

## Troubleshooting

### Out of Memory
- Reduce batch_size to 1
- Reduce lora_layers (4 instead of 16)
- Use gradient checkpointing: `--grad-checkpoint`
- Use 4-bit quantization

### Slow Training
- Increase batch_size if memory allows
- Reduce sequence length: `--max-seq-length 512`
- Use fewer LoRA layers

### Model Not Found
```bash
# Check available models
ls ~/.cache/huggingface/hub/

# Clear cache if needed
rm -rf ~/.cache/huggingface/hub/models--model--name
```

## Useful Aliases (.zshrc)
```bash
alias mlx-convert="mlx_lm.convert --hf-path"
alias mlx-chat="mlx_lm.chat --model"
alias mlx-gen="mlx_lm.generate --model"
alias mlx-train="mlx_lm.lora --train --model"
```

---
*Quick reference for MLX operations - keep this handy!*