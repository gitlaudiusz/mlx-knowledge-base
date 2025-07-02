# 💾 Memory Optimization Guide for MLX

## Understanding Apple Silicon Memory

### Unified Memory Architecture
- **Shared Pool**: CPU and GPU share the same memory
- **No Transfers**: Data doesn't need copying between CPU/GPU
- **Dynamic Allocation**: System manages memory automatically
- **Memory Pressure**: macOS will compress/swap when needed

### Memory Bandwidth by Chip
| Chip | Memory Bandwidth | Practical ML Limit |
|------|-----------------|-------------------|
| M1 | 200 GB/s | 7B models |
| M1 Pro | 200 GB/s | 13B models |
| M1 Max | 400 GB/s | 30B models |
| M1 Ultra | 800 GB/s | 65B models |
| M2 Series | Similar to M1 | Similar limits |
| M3 Max | 400 GB/s | 30B+ models |
| M3 Ultra | 800 GB/s | 180B+ models |

## Model Size Calculations

### Quick Reference
```
FP16: model_size = num_parameters × 2 bytes
INT8: model_size = num_parameters × 1 byte  
INT4: model_size = num_parameters × 0.5 bytes
```

### Real-World Examples
| Model | Parameters | FP16 | INT8 | INT4 |
|-------|-----------|------|------|------|
| Llama-7B | 7B | 14GB | 7GB | 3.5GB |
| Llama-13B | 13B | 26GB | 13GB | 6.5GB |
| Llama-30B | 30B | 60GB | 30GB | 15GB |
| Llama-70B | 70B | 140GB | 70GB | 35GB |
| Nemotron-253B | 253B | 506GB | 253GB | 126GB |

## Optimization Techniques

### 1. Quantization

#### Basic 4-bit Quantization
```bash
mlx_lm.convert --hf-path model --mlx-path output \
  --quantize --q-bits 4 --q-group-size 64
```

#### Aggressive Quantization
```python
# 2-bit for maximum compression (experimental)
mlx_lm.convert --hf-path model --mlx-path output \
  --quantize --q-bits 2 --q-group-size 32
```

### 2. LoRA Fine-tuning Optimizations

#### Minimal Memory Config (16GB)
```bash
mlx_lm.lora \
  --model 7b-model-4bit \
  --train \
  --batch-size 1 \
  --lora-layers 4 \
  --max-seq-length 512 \
  --grad-checkpoint
```

#### Gradient Checkpointing
- Trades compute for memory
- Reduces activation memory by ~30%
- Essential for large models
```bash
--grad-checkpoint  # Always use on limited RAM
```

#### Reduce LoRA Rank
```bash
# Lower rank = less memory
--lora-rank 8   # Minimal (vs default 16)
--lora-rank 32  # If you have headroom
```

### 3. Sequence Length Management

```bash
# Shorter sequences = less memory
--max-seq-length 512   # Minimum useful
--max-seq-length 1024  # Balanced
--max-seq-length 2048  # Default
--max-seq-length 4096  # Only with lots of RAM
```

### 4. Batch Size Optimization

```python
# Dynamic batch sizing based on available memory
import mlx.core as mx

def get_optimal_batch_size(model_size_gb, ram_gb):
    # Conservative formula
    free_ram = ram_gb - 8  # Reserve 8GB for system
    model_overhead = model_size_gb * 1.5  # Activation memory
    
    if free_ram > model_overhead * 2:
        return 4
    elif free_ram > model_overhead * 1.5:
        return 2
    else:
        return 1
```

## Advanced Memory Techniques

### 1. Model Sharding (Manual)
```python
# Load only specific layers
from mlx_lm import load

# Custom loading for huge models
def load_model_partial(path, start_layer=0, end_layer=32):
    # Implementation depends on model architecture
    pass
```

### 2. Activation Checkpointing
```python
# In training config
config = {
    "gradient_checkpointing": True,
    "checkpoint_layers": [8, 16, 24],  # Checkpoint every 8 layers
}
```

### 3. Memory Profiling
```bash
# Monitor memory during training
while true; do
  echo "$(date): $(ps aux | grep mlx | awk '{print $6/1024 " MB"}')"
  sleep 5
done
```

### 4. Swap Configuration (Last Resort)
```bash
# Increase swap (macOS)
sudo sysctl vm.swapusage  # Check current
# Note: macOS manages swap automatically
```

## Memory-Efficient Workflows

### For 16GB Macs
```bash
# 1. Use 4-bit models only
# 2. Batch size = 1
# 3. Gradient checkpointing ON
# 4. Max sequence = 512-1024
# 5. LoRA layers = 4-8

mlx_lm.lora \
  --model llama-7b-4bit \
  --batch-size 1 \
  --grad-checkpoint \
  --lora-layers 4 \
  --max-seq-length 512
```

### For 32GB Macs
```bash
# More flexibility
# 1. 4-bit for 13B, 8-bit for 7B
# 2. Batch size = 2-4
# 3. Sequence length = 1024-2048
# 4. LoRA layers = 8-16

mlx_lm.lora \
  --model llama-13b-4bit \
  --batch-size 2 \
  --grad-checkpoint \
  --lora-layers 8 \
  --max-seq-length 1024
```

### For 64GB+ Macs
```bash
# Run larger models
# 1. 30B models in 4-bit
# 2. Batch size = 4-8
# 3. Full sequence lengths
# 4. More LoRA layers

mlx_lm.lora \
  --model llama-30b-4bit \
  --batch-size 4 \
  --lora-layers 16 \
  --max-seq-length 2048
```

## Troubleshooting Memory Issues

### Signs of Memory Pressure
1. System becomes unresponsive
2. Colored memory pressure in Activity Monitor
3. Excessive swap usage
4. Kernel panics (extreme cases)

### Quick Fixes
```bash
# 1. Kill the process
pkill -f mlx

# 2. Clear cache
rm -rf ~/.cache/huggingface/hub/

# 3. Restart Terminal/Python
# 4. Reduce batch size to 1
# 5. Use more aggressive quantization
```

### Monitoring Tools
```bash
# Real-time memory usage
htop

# macOS Activity Monitor
open -a "Activity Monitor"

# Command line
vm_stat 1  # Virtual memory statistics

# MLX memory info
python -c "import mlx.core as mx; print(mx.metal.get_active_memory())"
```

## Best Practices

1. **Start Conservative**
   - Begin with minimal settings
   - Gradually increase if stable

2. **Monitor Actively**
   - Watch memory during first epoch
   - Note peak usage

3. **Leave Headroom**
   - Keep 2-4GB free for system
   - Avoid yellow/red memory pressure

4. **Profile Before Training**
   ```python
   # Test memory usage
   model, tokenizer = load("model-path")
   dummy_input = mx.zeros((1, 512))
   output = model(dummy_input)
   print(f"Memory used: {mx.metal.get_active_memory() / 1e9:.2f} GB")
   ```

5. **Use Efficient Data Loading**
   - Stream data instead of loading all
   - Use generators for large datasets

## Memory Optimization Checklist

- [ ] Quantized model to 4-bit or 8-bit
- [ ] Enabled gradient checkpointing
- [ ] Set batch size to 1 (increase if possible)
- [ ] Limited sequence length appropriately
- [ ] Reduced LoRA layers if needed
- [ ] Closed unnecessary applications
- [ ] Monitored memory usage
- [ ] Have swap space available
- [ ] Tested on small data first

---

*Remember: It's better to train slowly than to crash from OOM!*