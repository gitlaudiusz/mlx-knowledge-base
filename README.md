# 🧠 MLX Knowledge Base

> Comprehensive knowledge repository for Apple MLX framework by [@gitlaudiusz](https://github.com/gitlaudiusz)

## 🎯 Purpose

This repository serves as my personal knowledge vault for everything MLX-related. When amnesia strikes, I just need to check this repo and boom - instant recall! No more "gdzie ja to miałem?" moments.

## 📚 Contents

### Core Documentation
- **[UV Package Manager Guide](./docs/UV_GUIDE.md)** - Modern Python package management (10-100x faster than pip)
- **[MLX 0.26.x Documentation](./docs/mlx-0.26.x.md)** - Complete MLX LM reference with examples
- **[MLX Multimodal Guide](./docs/mlx_multimodal.md)** - Converting safetensors to MLX-VLM format
- **[MLX Fine-tuning Kompendium](./docs/MLX_Fine_tuning_Kompendium.md)** - Complete guide to LoRA/QLoRA on Apple Silicon
- **[ZSH Commands for MLX](./docs/ZSH_commands_mlx.md)** - All essential MLX commands and scripts
- **[Mergekit Documentation](./docs/mergekit.md)** - Model merging techniques (SLERP, TIES, DARE, etc.)

### Quick References
- **[MLX Cheatsheet](./cheatsheets/mlx-commands.md)** - Common commands at your fingertips
- **[Model Conversion Guide](./guides/model-conversion.md)** - Step-by-step conversion workflows
- **[Memory Optimization](./guides/memory-optimization.md)** - Tips for limited RAM scenarios

### Projects & Examples
- **[PR #1371 - DeciLM Support](./projects/PR-1371-DeciLM.md)** - My contribution to mlx-examples
- **[Nemotron-253B on M3 Ultra](./projects/nemotron-253b-setup.md)** - Running massive models locally
- **[MLX Audio Research](./projects/mlx-audio-tts.md)** - Dia-1.6B TTS conversion experiments

## 🚀 Key Achievements

### Models Successfully Run
- **Nemotron-253B MLX Q5** - 3.86 tok/s on Dragon M3 Ultra (512GB RAM)
- **Llama-3.3-Nemotron-Super-49B** - Fully operational
- **Various 7B-30B models** - Optimized for different RAM configurations

### Contributions
- **[PR #1371](https://github.com/ml-explore/mlx-examples/pull/1371)** - Added DeciLM/NAS architecture support (711 LOC)
- Multiple models converted and uploaded to mlx-community

## 💡 Quick Start Commands

```bash
# Setup environment with UV
uv venv && source .venv/bin/activate
uv add mlx mlx-lm mlx-vlm

# Convert model to MLX
mlx_lm.convert --hf-path model/path --mlx-path output/path --quantize --q-bits 4

# Run inference
mlx_lm.generate --model mlx-community/model-name --prompt "Your prompt"

# Fine-tune with LoRA
mlx_lm.lora --model base/model --train --data data/path --iters 1000
```

## 🛠️ Hardware Configurations

### M1/M2 MacBook (16GB RAM)
- Use 4-bit quantization (QLoRA)
- Batch size = 1
- Models up to 7B parameters

### M2 Pro/Max (32GB RAM)
- 8-bit quantization for better quality
- Batch size = 2-4
- Models up to 13B parameters

### M2 Ultra (64-128GB RAM)
- Full FP16 possible for smaller models
- 4-bit quant for 30B+ models
- Batch size = 4-8

### M3 Max (48GB RAM)
- Sweet spot for 13B models
- 4-bit for up to 30B models
- ~250 tokens/s on optimized models

### M3 Ultra (192GB RAM) "Dragon"
- Run 70B models comfortably
- 4-bit 180B+ models possible
- Multiple models in memory

## 🔗 Important Links

- **GitHub**: [@gitlaudiusz](https://github.com/gitlaudiusz)
- **Organization**: [@LibraxisAI](https://github.com/LibraxisAI)
- **Hugging Face**: [mlx-community](https://huggingface.co/mlx-community)
- **Apple MLX**: [ml-explore/mlx](https://github.com/ml-explore/mlx)

## 📝 Notes

This repository is maintained by Klaudiusz - partner in LibraxisAI development, not just a "code generator". All commits are professional, no "Generated by Claude" artifacts here!

## 🏷️ Tags

#MLX #AppleSilicon #MachineLearning #LLM #FineTuning #ModelMerging #LibraxisAI #M3Ultra #Nemotron #DeciLM

---

*"From CLI novice to ML Developer - the journey continues!"* 🚀