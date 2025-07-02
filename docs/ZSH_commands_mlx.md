
# Bash Commands Mlx 
- aktualne dla wersji `mlx>=0.26.x`.

### Running DeepSeek AI's model with MLX LM
```zsh
mlx_lm.chat --model mlx-community/DeepSeek-V3-0324-4bit
```

### Text generation with MLX LM
```
mlx_lm.generate --model "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
                --prompt "Write a quick sort in Swift"
```
### Changing the model's behavior with flags
```zsh
mlx_lm.generate --model "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
                --prompt "Write a quick sort in Swift" \
                --top-p 0.5 \
                --temp 0.2 \
                --max-tokens 1024
```
### Getting help for MLX LM
```
mlx_lm.generate --help
```
## MLX LM Python API
### Using MLX LM from Python
```Python
from mlx_lm import load, generate

# Load the model and tokenizer directly from HF
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# Prepare the prompt for the model
prompt = "Write a quick sort in Swift"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

# Generate the text
text = generate(model, tokenizer, prompt=prompt, verbose=True)
```
### Inspecting model architecture
```
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

print(model)
print(model.parameters())
print(model.layers[0].self_attn)
```
### Generation with KV cache
```python
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache

# Load the model and tokenizer directly from HF
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# Prepare the prompt for the model
prompt = "Write a quick sort in Swift"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

cache = make_prompt_cache(model)

# Generate the text
text = generate(model, tokenizer, prompt=prompt, prompt_cache=cache, verbose=True)
```
### Quantization
```zsh
mlx_lm.convert --hf-path "mistralai/Mistral-7B-Instruct-v0.3" \
               --mlx-path "./mistral-7b-v0.3-4bit" \
               --dtype float16 \
               --quantize --q-bits 4 --q-group-size 64
```
### Model quantization with MLX LM in Python
```Python
from mlx_lm.convert import convert

# We can choose a different quantization per layer
def mixed_quantization(layer_path, layer, model_config):
    if "lm_head" in layer_path or "embed_tokens" in layer_path:
        return {"bits": 6, "group_size": 64}
    elif hasattr(layer, "to_quantized"):
        return {"bits": 4, "group_size": 64}
    else:
        return False

# Convert can be used to change precision, quantize and upload models to HF
convert(
    hf_path="mistralai/Mistral-7B-Instruct-v0.3",
    mlx_path="./mistral-7b-v0.3-mixed-4-6-bit",
    quantize=True,
    quant_predicate=mixed_quantization
)
```
### Model fine-tuning
```zsh
mlx_lm.lora --model "mlx-community/Mistral-7B-Instruct-v0.3-4bit" --train --data /path/to/our/data/folder --iters 300 --batch-size 16
```
### Prompting before fine-tuning
```zsh
mlx_lm.generate --model "./mistral-7b-v0.3-4bit" \
    --prompt "Who won the latest super bowl?"
```
### Fine-tuning to learn new knowledge
```zsh
mlx_lm.lora --model "./mistral-7b-v0.3-4bit" 
						--train 
            --data ./data 
            --iters 300 
            --batch-size 8 
            --mask-prompt 
            --learning-rate 1e-5
```
### Prompting after fine-tuning
```zsh
mlx_lm.generate --model "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
                --prompt "Who won the latest super bowl?" \
                --adapter "adapters"
```
### Fusing models
```zsh
mlx_lm.fuse --model "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
            --adapter-path "path/to/trained/adapters" \
            --save-path "fused-mistral-7b-v0.3-4bit" \
            --upload-repo "my-name/fused-mistral-7b-v0.3-4bit"
```            
### Fusing our fine-tuned model adapters
```zsh
mlx_lm.fuse --model "./mistral-7b-v0.3-4bit" \
            --adapter-path "adapters" \
            --save-path "fused-mistral-7b-v0.3-4bit"
```
### LLMs in MLX Swift
```
import Foundation
import MLX
import MLXLMCommon
import MLXLLM

@main
struct LLM {
    static func main() async throws {
        // Load the model and tokenizer directly from HF
        let modelId = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        let modelFactory = LLMModelFactory.shared
        let configuration = ModelConfiguration(id: modelId)
        let model = try await modelFactory.loadContainer(configuration: configuration)
        
        try await model.perform({context in
            // Prepare the prompt for the model
            let prompt = "Write a quicksort in Swift"
            let input = try await context.processor.prepare(input: UserInput(prompt: prompt))
            
            // Generate the text
            let params = GenerateParameters(temperature: 0.0)
            let tokenStream = try generate(input: input, parameters: params, context: context)
            for await part in tokenStream {
                print(part.chunk ?? "", terminator: "")
            }
        })
    }
}
```
### Generation with KV cache in MLX Swift
```Python
import Foundation
import MLX
import MLXLMCommon
import MLXLLM

@main
struct LLM {
    static func main() async throws {
        // Load the model and tokenizer directly from HF
        let modelId = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        let modelFactory = LLMModelFactory.shared
        let configuration = ModelConfiguration(id: modelId)
        let model = try await modelFactory.loadContainer(configuration: configuration)
        
        try await model.perform({context in
            // Prepare the prompt for the model
            let prompt = "Write a quicksort in Swift"
            let input = try await context.processor.prepare(input: UserInput(prompt: prompt))

            // Create the key-value cache
            let generateParameters = GenerateParameters()
            let cache = context.model.newCache(parameters: generateParameters)

            // Low level token iterator
            let tokenIter = try TokenIterator(input: input,
                                              model: context.model,
                                              cache: cache,
                                              parameters: generateParameters)
            let tokenStream = generate(input: input, context: context, iterator: tokenIter)
            for await part in tokenStream {
                print(part.chunk ?? "", terminator: "")
            }
        })
    }
}
```



# Bash Commands Mlx 
- aktualne dla wersji `mlx<=0.25.x`.

## Table of Contents

- [🔑 Dostępne API i klucze](#-dostępne-api-i-klucze)
- [📜 Dostępne skrypty](#-dostępne-skrypty)
  - [1. Konwersja z Claude (vet-to-mlx.py)](#1-konwersja-z-claude-vet-to-mlxpy)
  - [2. Konwersja z DeepSeek (deepseek-to-mlx.py)](#2-konwersja-z-deepseek-deepseek-to-mlxpy)
  - [3. Konwersja z LM Studio (lmstudio-to-mlx.py)](#3-konwersja-z-lm-studio-lmstudio-to-mlxpy)
- [🚀 Proces fine-tuningu MLX-LM](#-proces-fine-tuningu-mlx-lm)
  - [1. Konwersja modelu do MLX](#1-konwersja-modelu-do-mlx)
  - [2. Fine-tuning z LoRA](#2-fine-tuning-z-lora)
  - [3. Fuzja adaptera z modelem](#3-fuzja-adaptera-z-modelem)
  - [4. Testowanie modelu](#4-testowanie-modelu)
- [📋 Formatowanie danych](#-formatowanie-danych)
- [💡 Wskazówki i najlepsze praktyki](#-wskazówki-i-najlepsze-praktyki)


# Podsumowanie działających poleceń
## WAŻNE
MLX ma zapewnioną pełną kompatybilność TYLKO z python'em w wersji **3.11.x**.
Przed instalacją upewnij się, że używasz odpowiedniej wersji PYTHON'a.
```bash
python --version # lub python3 --version
```
W przypadku posiadania globalnie innej wersji, zastosuj `pyenv` dla zarządzania wersją PYTHON'a w projekcie:
```
brew install pyenv
```
po instalacji upewnij się, że dodano do pliku konfiguracyjnego powłoki, np. `.zshrc` następujące linie:
```
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```
zrestartuj terminal lub odśwież powłokę poleceniem 
```bash
source ~/.zshrc
```
Sprawdź czy pyenv zainstalował się poprawnie poleceniem 
```bash
pyenv --version && which pyenv
```
Zainstaluj uvicorn'a jeśli nie masz (xD):
```bash
$ pip install uvicorn
```
Podjedź do mięsnego :xD: :
```
# 1. Tworzenie środowiska wirtualnego i instalacja pakietów
uv venv                                      # Utworzenie środowiska wirtualnego
source .venv/bin/activate                    # Aktywacja środowiska
uv pip install mlx mlx-lm mlx-vlm mlx-audio huggingface_hub hf_transfer  # Instalacja pakietów

huggingface-cli login
# Hugging Face Token (from .env file)
# Load using: source .env && echo $HF_TOKEN

# 2. Pobranie i konwersja modeli

cd models/gemma-3-27b-it      
huggingface-cli download google/gemma-3-27b-it  model.safetensors.index.json --local-dir ./                             
huggingface-cli download google/gemma-3-27b-it --local-dir ./
cd ../..

python3 convert_simple.py --input-dir models/bielik --output-dir models/Bielik-2.3-MLX --force  # Konwersja do FP16
python3 convert_simple.py --input-dir models/bielik --output-dir models/Bielik-2.3-MLX-8bit --quantize 8 --q-group-size 128 --force  # Kwantyzacja 8-bit
python3 convert_simple.py --input-dir models/pllum-12b --output-dir models/PLLuM-12B-MLX-8bit --quantize 8 --q-group-size 128 --force  # Kwantyzacja modelu PLLuM

# Konwersja z użyciem MLX-VLM:

python convert_mlx-vlm.py \
    --input-dir models/gemma-3-27b-it \
    --output-dir models/gemma-3-27b-it-mlx \
    --model-type "gemma-3" \
    --quantize 8 \
    --q-group-size 128 \
    --precision fp16 \
    --max-seq-len 8192 \
    --attention-impl flash \
    --verbose

# 3. Testowanie modeli
python3 -m mlx_lm.generate --model models/Bielik-2.3-MLX-8bit --prompt "Wyjaśnij, jak działa ketamina..." --max-tokens -1
python3 -m mlx_lm.generate --model models/PLLuM-12B-MLX-8bit --prompt "Wyjaśnij, jak działa ketamina..." --max-tokens -1
python3 -m mlx_lm.generate --model models/eskulap-alpha-1-MLX --prompt "Co to jest żaneta w kontekście medycznym?" --max-tokens 200

# 4. Pobieranie modeli
mkdir -p models/gemma-3-27b-it
cd models/gemma-3-27b-it
huggingface-cli download google/gemma-3-27b-it --local-dir ./ -f model.safetensors.index.json
huggingface-cli download google/gemma-3-27b-it --local-dir ./
cd ../..
```

Dodatkowo:
```
# Kompleksowy przewodnik do fine-tuningu z reasoning traces

## 🔑 Dostępne API i klucze

| Model | Typ | API Key |
| ----- | --- | ------- |
| Claude 3.7 Sonnet | Anthropic | `$CLAUDE_API_KEY` (from .env) |
| DeepSeek R1 | DeepSeek | `$DEEPSEEK_API_KEY` (from .env) |
| QwQ-32B | LM Studio | Nie wymagany (lokalny) |

## 📜 Dostępne skrypty

### 1. Konwersja z Claude (vet-to-mlx.py)

```bash
# Load the API key from .env file
python vet-to-mlx.py --input hemodializa.json --output-dir ready/data/hemodializa --api-key "$CLAUDE_API_KEY"
```

### 2. Konwersja z DeepSeek (deepseek-to-mlx.py)

```bash
# Load the API key from .env file
python deepseek-to-mlx.py --input hemodializa.json --output-dir ready/data/hemodializa --api-key "$DEEPSEEK_API_KEY"
```

### 3. Konwersja z LM Studio (lmstudio-to-mlx.py)

```bash
# Dla lokalnego serwera
python lmstudio-to-mlx.py --input hemodializa.json --output-dir ready/data/hemodializa --server http://localhost:1234

# Dla serwera w sieci
python lmstudio-to-mlx.py --input hemodializa.json --output-dir ready/data/hemodializa --server http://zero.libraxis.cloud:1234
```

## 🚀 Proces fine-tuningu MLX-LM

### 1. Konwersja modelu do MLX

```bash
# Konwersja do formatu MLX FP16
python -m mlx_lm.convert --hf-path google/gemma-3-27b-it --output-dir models/google/gemma-3-27b-MLX

# Kwantyzacja do 8-bit dla szybszej inferencji
python -m mlx_lm.convert --hf-path models/bielik --output-dir models/Bielik-2.3-MLX-8bit --quantize 8 --q-group-size 128

# Kwantyzacja do 4-bit dla QLoRA (M3 Max)
python -m mlx_lm.convert --hf-path models/bielik --output-dir models/Bielik-2.3-MLX-4bit --quantize 4 --q-group-size 128
```

### 2. Fine-tuning z LoRA

```bash
# Fine-tuning z LoRA na M2 Ultra
python -m mlx_lm.lora --model models/Bielik-2.3-MLX-FP16 \
    --train --data ready/data/hemodializa \
    --batch-size 2 --seq-len 1024 \
    --iters 1000 \
    --lr 2e-4 \
    --lora-rank 16 --lora-alpha 32 \
    --grad-checkpoint \
    --save-every 200 \
    --adapter-path adapters/hemodializa/adapter.safetensors

# Fine-tuning z QLoRA na M3 Max
python -m mlx_lm.lora --model models/Bielik-2.3-MLX-4bit \
    --train --data ready/data/hemodializa \
    --batch-size 1 --seq-len 768 \
    --iters 1000 \
    --lr 2e-4 \
    --lora-rank 16 --lora-alpha 32 \
    --grad-checkpoint \
    --adapter-path adapters/hemodializa/adapter.safetensors
```

### 3. Fuzja adaptera z modelem

```bash
python -c "
import os
import mlx.core as mx
from mlx_lm import load, save
from pathlib import Path

# Ładowanie modelu i adaptera
model_path = Path('models/Bielik-2.3-MLX-FP16')
adapter_path = Path('adapters/hemodializa/adapter.safetensors')
output_path = Path('models/Bielik-Hemodializa-MLX')

# Utworzenie katalogu wyjściowego
os.makedirs(output_path, exist_ok=True)

# Wczytanie modelu i adaptera
print('Wczytywanie modelu bazowego...')
model, tokenizer = load(model_path.as_posix())

print('Wczytywanie adaptera LoRA...')
adapter_weights = mx.load(adapter_path.as_posix())

# Fuzja adaptera z modelem
print('Fuzja adaptera z modelem...')
model.update_weights(adapter_weights)

# Zapisanie sfuzowanego modelu
print('Zapisywanie sfuzowanego modelu...')
save(output_path.as_posix(), model, tokenizer)

print(f'Model sfuzowany pomyślnie i zapisany w {output_path}')
"
```

### 4. Testowanie modelu

```bash
python -m mlx_lm.generate \
    --model models/Bielik-Hemodializa-MLX \
    --prompt "Zadanie: Wyjaśnij rolę ultradźwięków w terapii hemodializą.\nOdpowiedź:" \
    --max-tokens 512 \
    --temp 0.1
```

## 📋 Formatowanie danych

Dane wejściowe są oczekiwane w formacie:

```json
{"instruction": "Podaj definicję terminu 'hemodializa'.", 
 "input": "Artykuł opisuje hemodializę jako procedurę terapeutyczną stosowaną u psów.", 
 "output": "Hemodializa to procedura oczyszczania krwi."}
```

Po konwersji format MLX-LM to:

```json
{"prompt": "Zadanie: Podaj definicję terminu 'hemodializa'.\nArtykuł opisuje hemodializę jako procedurę terapeutyczną stosowaną u psów.\nOdpowiedź:", 
 "completion": "<think>rozumowanie</think>\nHemodializa to procedura oczyszczania krwi."}
```

## 💡 Wskazówki i najlepsze praktyki

1. **Wybór skryptu do reasoning traces**:
   - Claude 3.7 Sonnet oferuje najwyższą jakość reasoning
   - DeepSeek R1 jest szybszy, ale czasem generuje błędy
   - LM Studio z QwQ-32B jest najekonomiczniejszy (lokalny), ale wolniejszy

2. **Optymalizacja dla sprzętu**:
   - M2 Ultra (128GB): FP16 z batch_size=2
   - M3 Max (48GB): 4-bit QLoRA z batch_size=1
   - Zawsze używaj `--grad-checkpoint` dla oszczędzania pamięci

3. **Typowe błędy i rozwiązania**:
   - Problemy z parsowaniem JSON: Spróbuj alternatywnego formatu pliku
   - Błędy API: Zwiększ opóźnienia między zapytaniami
   - Problemy z pamięcią: Zmniejsz batch_size lub seq_len

Szczegółowe informacje o procesie konwersji i fine-tuningu znajdziesz w dokumentach "instrukcja basic mlx-lm + Lora" oraz "sposoby optymalizacji dla MLX-LM".
```

Formatowanie jako skróty do `.zshrc`:

```bash
# Skróty do dodania w .zshrc
alias mlx-venv="uv venv && source .venv/bin/activate"
alias mlx-install="uv pip install mlx mlx-lm huggingface_hub hf_transfer"
alias mlx-convert-fp16="python3 convert_simple.py --input-dir \$1 --output-dir \$2 --force"
alias mlx-convert-8bit="python3 convert_simple.py --input-dir \$1 --output-dir \$2 --quantize 8 --q-group-size 128 --force"
alias mlx-convert-4bit="python3 convert_simple.py --input-dir \$1 --output-dir \$2 --quantize 4 --q-group-size 128 --force"
alias mlx-generate="python3 -m mlx_lm.generate --model \$1 --prompt \"\$2\" --max-tokens \$3"
alias hf-download="huggingface-cli download \$1 --local-dir ./ && huggingface-cli download \$1 --local-dir ./ -f model.safetensors.index.json"
```
Deepseek to MLX
```
# Load the API key from .env file
python deepseek-to-mlx.py --input A.vasorumjsonl2.json --output-dir ready/data/A.vasorum --api-key "$DEEPSEEK_API_KEY" --workers 6
```

Problemy z MLX-VLM i wymuszenie python 3.11.11 z racji na kompatybilność ze scipy==1.13.1

⏺ Bash(
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

```
⏺ Teraz zainstalujmy Python 3.11 przez pyenv:

⏺ Bash
```
export PYENV_ROOT="$HOME/.pyenv"  …
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install 3.11)

```
⏺ Ustawmy tę wersję Pythona dla naszego projektu:

  Bash(cd /Users/sztudio/mlx-finetuning  …

⏺ Bash
```
cd /Users/sztudio/mlx-finetuning  …
      export PYENV_ROOT="$HOME/.pyenv"
      export PATH="$PYENV_ROOT/bin:$PATH"
      eval "$(pyenv init -)"
      pyenv local 3.11.11
      python --version)
  ⎿  Python 3.11.11
```
⏺ Teraz stwórzmy nowe środowisko wirtualne z Pythonem 3.11 i zainstalujmy scipy 1.13.1:

⏺ Bash 
```
cd /Users/sztudio/mlx-finetuning  …
      export PYENV_ROOT="$HOME/.pyenv"
      export PATH="$PYENV_ROOT/bin:$PATH"
      eval "$(pyenv init -)"
      python -m venv .venv-py311
      source .venv-py311/bin/activate
      python --version
      pip install scipy==1.13.1)
```
⏺ Bash
```
cd /Users/sztudio/mlx-finetuning…
      source .venv-py311/bin/activate
      pip install mlx-vlm)
```