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

## Webinar transcript:
```markdown
Hi, I’m Angelos, an engineer in the MLX team.
 Today, I’ll show you how MLX is perfect for large-language models on Apple Silicon.
 With it, you can perform inference and fine-tune massive models right from your Mac.
 And you can do all this with CLI applications or from Python or Swift.
 If you are new to MLX, it is an open-source library that is purpose-built for doing machine learning on Apple Silicon.
 It utilizes Metal for acceleration on the GPU and takes advantage of unified memory so that operations on the CPU and GPU can work on the same data simultaneously.
 You can use MLX in your favorite language since it provides APIs in Python, Swift, C++ and even C.
 To learn more check out our session “Get Started with MLX for Apple Silicon”.
 When it comes to running large language models on on Apple Silicon, MLX unlocks powerful new capabilities, allowing you to run the latest state-of-the-art models right on your Mac with a single-line command.
 Let’s load DeepSeek AI’s latest model, which has an impressive 670 billion parameters.
Even when quantized to 4.
5-bits per weight, the model’s weights alone still require around 380 gigabytes of memory.
 To handle that, we're using an M3 Ultra with its massive 512 gigabytes of unified memory, no other consumer device comes close.
 Now that the model is loaded, we can start interacting with it.
 We can ask it questions like, “What is the deepest lake in the United States?” Or have it write code for us.
As you can see, MLX enables smooth, real-time interaction and generation at faster than reading speeds, even with models containing hundreds of billions of parameters, all running locally right on your Mac desktop.
 Now that you've seen what's possible, let's dive into how you can use MLX to run these powerful models on your own Mac.
 We will start by introducing MLX LM, a Python library and a set of command line applications that can address all of your large language model requirements, providing a robust and versatile solution for a wide range of applications.
Subsequently, we will delve into text generation with MLX LM and show how easy it is to generate text either from Python or from the terminal.
 In addition, we will go over downloading models from Hugging Face and quantizing them for faster inference on device.
MLX can do much more than just inference, though.
 So next, we will use MLX LM to fine tune a language model on our own data.
 In particular, we will train a low-rank adapter, which we can then fuse into the model for easier deployment and faster inference.
Finally, we will go over using MLX from Swift, where we will see how you can integrate a large language model in your Swift application with only a few lines of code.
The easiest way to get started with language models in MLX is by using MLX LM.
 MLX LM is a Python package built on top of MLX, designed for running and experimenting with large language models.
 It provides a set of command line tools that let you generate text or fine-tune models all without writing any code.
 And if you do want more control, it also provides a Python API so you can customize the generation or training process as much as you like.
 It’s also tightly integrated with Hugging Face.
 That means you can quickly download thousands of models from the internet and even upload your own to share with the community.
Getting started is easy; just run pip install mlx-lm.
Let's now delve into the details for the most common use case for language models: generating text.
This is a command line tool that lets you generate text using a language model, right from your terminal, no code required.
 Here’s how it works; you give it a model from Hugging Face or a local path, a text prompt, and it handles the rest.
 It downloads the model if needed, it runs the prompt through it, and prints the generated response.
 So instead of just talking about it, let’s run this command.
Within just a few seconds, we get a Swift implementation of Quick Sort.
You can tweak the behavior of the model by adding flags for things like sampling temperature, top-p or max tokens, just like with any standard text generation setup.
 And if you’re curious about all the available options, you can always run mlx_lm.
generate --help So whether you’re prototyping ideas, generating code, or just exploring what the model can do, this is the simplest place to start.
 We just saw how easy it is to generate text from the command line using mlx_lm.
generate.
 But one of the real strengths of MLX LM is that it's not limited to terminal tools.
 It also provides a clean and flexible Python API, perfect when you want more fine-grained control or need to integrate generation into a larger workflow.
 Let’s take a look at how we can do the same thing, generating text, using just a few lines of Python.
First, we import two utilities; load and generate.
 Load, as the name suggests, handles everything related to model loading.
 It fetches the requested model, either from your local disk or directly from Hugging Face and sets up the model object along with the tokenizer.
 Then we call generate.
 This function performs a token generation loop and returns the output text, which we can process further in Python, log, or feed into other systems.
So with just these two steps, load, then generate, we get the same functionality as the CLI, but with full control and flexibility in Python.
 So here’s another powerful aspect of MLX LM’s Python API.
 The model you get from load isn’t some opaque object you can only interact with through a fixed interface.
 It’s a fully structured MLX neural network, which means you can inspect it, explore its architecture, and even modify it.
 Let me show you a quick demo.
We can start by printing the list of layers that make up the model.
This gives us a full breakdown of the transformer stack, layer by layer.
 We can also take a look at the model’s parameters, essentially the weights and biases that the model has learned.
And if we want to dig into a specific part of the network, say the self-attention module in the first layer, we can do that too.
This level of transparency is really useful, not just for debugging or learning, but also if you want to experiment with things like layer swapping, custom fine-tuning routines, or low-level model surgery.
So far, we’ve seen how to generate text from a single prompt.
 But what if you want to maintain a conversation, or generate responses in multiple turns where each new prompt builds on the previous one? That’s where the key value cache, or KV cache, comes in.
 Language models use attention mechanisms to process input tokens and during generation they repeatedly compute attention over all previously generated tokens.
 This can get expensive, especially for long prompts or multi-turn scenarios.
 A KV cache solves this by storing intermediate results from earlier steps, specifically the keys and values.
Instead of recomputing everything from scratch, the model reuses this cache, saving time and computation.
 In MLX LM, using a KV cache is straightforward.
Let’s update the previous Python example with an explicitly created KV cache that we can reuse for multiple generations.
We first create the cache object using the make_prompt_cache function.
 We can use it to edit the history in place, save it for later usage, or swap between conversations seamlessly.
Then, we pass it into the generate function.
 And as new tokens are generated, the cache gets updated.
 Each call continues from where the last one left off, maintaining context across turns.
 This is especially useful when building chatbots, virtual assistants, or any interactive application where keeping track of history matters.
Now let’s switch gears a bit and talk about model quantization.
 We’ve seen how to generate text and work with models interactively.
 But for real-world deployment, efficiency becomes just as important as functionality.
 Models are usually released in the same precision they were trained with, like float32 or float16.
 That is accurate, but it makes them large and slow, especially on smaller devices.
 That's where quantization comes in.
It reduces the model to lower precision, like Int8 or even 4-bit, which reduces memory use and speeds up inference, often with little impact on quality.
 But usually, quantization involves extra tools, conversion scripts, and compatibility headaches.
 In MLX, it's much simpler.
 Quantization is built-in.
 You can compress models at various levels and use them right away for inference or training with no extra setup.
 Let's take a look at how this works.
To quantize, or generally convert a model with MLX, you use the mlx_lm.
convert command.
 This tool takes care of downloading a model from Hugging Face, converting it to a different precision, and saving it locally all in one step.
In this example, we’re fetching the original 16-bit Mistral model and quantizing it to around 4-bits per weight.
The result is a significantly smaller model that’s faster to run and requires less memory.
 Once converted, the model is saved to the specified folder and can be used immediately for inference or training using the same MLX LM tools.
And if you want to share your quantize model with others, you can easily upload it back to Hugging Face by passing in a repository name.
 So whether you’re optimizing for speed, saving space, or contributing back to the community, this one command is all you need.
Just like with text generation, using the Python API to convert and quantize models gives you more flexibility without adding complexity.
 In fact, MLX LM makes it easy to apply different quantization settings to different parts of the model or from Python.
For example, it’s common practice to keep the embedding and final projection layers in higher precision since they tend to be more sensitive to quantization.
 In this example, we quantize those layers to 6-bits while the rest of the model uses 4-bits, striking a great balance between quality and efficiency.
 This is done by passing a quantization predicate function, a small function that receives each layer and returns the quantization parameters to use for it.
 Everything else works exactly the same.
 We call convert, pass the Hugging Face path and local output directory, and MLX handles the rest, including downloading the model and saving the quantized result.
 This fine-grained control is especially useful when you’re experimenting with model compression or trying to find the best trade-off between performance and accuracy.
So far, we've seen how to generate text using large language models and how to quantize them for faster inference and lighter deployment.
 But MLX can do more, especially when it comes to training.
 With MLX LM, you can fine-tune a large language model on your own data right on your Mac, and crucially, without that data ever leaving your device.
 And the best part, you can do it without writing a single line of code.
 Let’s take a look at how fine-tuning works.
Large language models are usually trained on massive, general-purpose datasets from across the Internet.
 That gives them broad knowledge, but it also means they might lack depth in specialized domains or miss the tone and language of a specific task.
 Fine-tuning is how we adapt these models to new contexts.
 By training them further on a smaller, domain-specific dataset, we can give them new capabilities or tailor their responses to particular needs.
 Traditionally, this process is done in the cloud, which can be expensive and often not ideal when you’re working with private or sensitive data.
 But with MLX, you can fine-tune large language models locally on your Mac, no cloud required, and no data ever leaves your machine.
 It is efficient, secure, and seamlessly integrated into the MLX workflow.
MLX LM supports two types of fine-tuning out of the box: full model fine-tuning and low-rank adapter training.
 In full fine-tuning, we update all the parameters of the pre-trained model.
 This gives you maximum flexibility, but it's also more resource intensive.
 In contrast, adapter training, specifically low-rank adapters, adds a small number of new parameters to the model and trains only those, while keeping the original network frozen.
 This makes training faster, lighter, and often more memory efficient, especially on local hardware.
 Let’s look at how we can apply this in practice by fine-tuning the Mistral model on a custom dataset.
 Let’s take a look at how easy it is to launch a fine-tuning job with MLX LM.
 It only takes a single command and just a few key arguments.
 We specify the model we want to fine-tune, the path to the dataset, and how long we want to train.
 Because quantization is deeply integrated into MLX, the mlx_lm.
lora command can even train adapters on top of quantized models.
 This dramatically reduces memory usage without sacrificing the ability to fine-tune effectively.
In this example, we're training on a 4-bit quantized version of Mistral, which cuts memory usage for the model weights by about 3.
5 times compared to the full precision version.
 So even with large models, fine-tuning remains practical and efficient right on your Mac.
That single line command is perfect for a quick training run, especially when you’re just getting started.
 But if you want to really fine tune performance, you’ll likely need more control over the training process.
 That’s where the training configuration file comes in.
 MLX LM supports config files that give you fine-grained control over all aspects of training, including path size, learning rate schedules, optimizer settings, evaluation intervals, and more.
 This lets you tailor the training setup to your specific data set, hardware, or optimization goals, and get the most out of your adapter.
 Let’s now see fine-tuning in action and how it can update a model’s knowledge.
 We start by asking Mistral 7b who won the latest Super Bowl.
As expected, the answer is correct, but outdated.
 The model’s knowledge cutoff means it doesn’t have access to recent events.
 But the beauty of fine tuning is that we can fix this in just a few minutes.
 By training on a small dataset with questions and answers about the latest Super Bowl, we can update the model’s knowledge and have it answer accurately.
After just a few minutes of fine-tuning, the model is now able to respond with up-to-date answers about teams, players, scores and more.
Now that we've trained our adapters, we can use MLX LM to fuse them back into the base model.
 This is especially useful for deployment and sharing because it produces a single self-contained model that's easy to distribute and use.
The fusion process combines the adapter with the original weights, resulting in a model that has the same architecture and number of parameters as the pre-trained version, just with updated capabilities.
 So from the outside it behaves like any other model, but with your fine-tuned knowledge built in.
To fuse the adapter into the model, we use the mlx_lm.
fuse command.
 It computes the fused weights and saves the results to the specified path, all in one step.
 There’s no need to manually dequantize or requantize anything.
 MLX handles that automatically and preserves the same quantization used during training.
 And if you want to share your newly fine-tuned model with others, it’s just as easy.
 You simply provide a Huggin Face repository name and the fused model will be uploaded and ready to use.
 So far, we’ve used Python to generate text fine-tune large language models.
 But one of MLX’s standout features is that it brings the same simplicity and flexibility to Swift.
 Let’s take a look at just how easy it is to use a large language model in Swift with MLX.
Here’s a complete example of how to load a quantized Mistral model and generate text all from Swift.
 And the entire thing fits in just 28 lines of code.
 We start by importing MLX and the language model libraries.
 Then we create a model container, an actor that safely manages concurrent access to the model and tokenizer.
 Next, we prepare the input.
 We tokenize the prompt, converting it into the numerical format the model understands.
 Finally, we run the generation loop and print the result, just like we saw earlier in Python.
 It’s the same workflow, the same capabilities, but now fully native in Swift.
 Let’s now see what it takes to retain the history of a conversation across multiple interactions with a model, just like we did in Python earlier.
 In Swift, this requires just a few extra lines.
The key idea is the same, we need to explicitly create a key value cache so we can reuse it across multiple generations.
 This is done with a single additional line of code.
 No complexity added.
 To manage the interaction more precisely, we also use a token iterator, which allows us to set the key value cast directly and control generations step by step.
 This setup gives us the flexibility to handle multi-turn conversations and advanced prompting, all from Swift.
 Throughout this session, we’ve seen just how simple it is to perform inference, training, and quantization with MLX, whether through code or terminal commands.
 Everything we’ve used, from the higher-level language model APIs down to the Metal kernels that power them, is fully open-source.
 MLX provides core operations in C, C++, Python, and Swift, with high-level APIs in Python and Swift, giving you both flexibility and control across the entire stack.
 This makes MLX uniquely powerful for running language models and machine learning workflows on Apple hardware.
 Let’s now take a look at where you can go from here.
 We’ve explored some of the key features of MLX LM, but there’s much more you can do.
 Our documentation dives deeper into advanced features like distributed inference and training, learned quantization, and custom training loops.
 To get hands-on quickly, the MLX and MLX Swift example repositories offer ready-to-run projects for tasks like image generation with diffusion models, speech recognition, and full language model training.
 Whether you’re building your own AI application or exploring under the hood, everything you need is just a few clicks away to get started.
 We can’t wait to see the amazing experiences you will create on Apple hardware using MLX and the power of large language models.
```