# 🛡️ adversarial-codegen

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: Active](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/yourusername/adversarial-codegen)

This repository evaluates the robustness of Large Language Models (LLMs) under various adversarial attacks, focusing on code generation tasks. We test both original and compressed LLMs across different datasets to provide comprehensive insights into model vulnerabilities.

## 📊 Overview
Our framework provides:
- 🔨 Implementation of various adversarial attack methods for code generation
- 🤖 Support for multiple LLM architectures (both original and compressed)
- 📈 Extensive evaluation across diverse coding datasets
- 🎯 Standardized benchmarking and comparison tools

## 🚀 Supported Models (In Plan)
We will support both original LLMs and their compressed versions.

- <img src="./assets/llama_logo.jpg" width="20"> [CodeLLaMA](https://github.com/meta-llama/codellama)  <!-- Download from Meta's official repo --> :heavy_check_mark:
- <img src="./assets/starcoder_logo.png" width="20"> [StarCoder](https://github.com/bigcode-project/starcoder)  <!-- From BigCode/HuggingFace --> :muscle:
- <img src="./assets/codegen_logo.png" width="20"> [CodeGen](https://github.com/salesforce/CodeGen)     <!-- From Salesforce --> :muscle:
- <img src="./assets/deepseek_logo.png" width="20"> [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder-V2)   <!-- From DeepSeek official website --> :heavy_check_mark:
- <img src="./assets/incoder_logo.png" width="20"> [InCoder](https://github.com/dpfried/incoder)     <!-- From Meta/Facebook --> :muscle:
- 🎩 [Magicoder](https://github.com/ise-uiuc/magicoder) :muscle:

## 🛠️ Implemented Attack Methods (In Plan)
1. 🎯 Natural Noise Injection
   - ⌨️ Typos and character swaps
   - 📝 Spacing and formatting variations
   - 💭 Comment modifications

2. 🏗️ Structural Attacks
   - 🔄 Variable name perturbations
   - 🔀 Control flow modifications
   - 🔌 API usage variations

3. 🔄 Semantic Preserving Transformations
   - 🔧 Code refactoring
   - 🔁 Equivalent syntax modifications
   - 🧮 Logic preservation with structural changes

## 📚 Datasets
- 👥 HumanEval / HumanEval Plus
- 📘 MBPP / MBPP Plus

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/adversarial-codegen
cd adversarial-codegen

# Install the package
pip install -e .
```

## 🎮 Usage
After installation, you can use the main functionality through the command-line interface:
```bash
adversarial-codegen attack [OPTIONS]
```
### 🔑 Required Arguments

- model_path: 📂 Path to the original model
- save_prompts: 💾 Directory path to save generated prompts
- save_results: 📊 Directory path to save attack results

### ⚡ Optional Arguments

#### 🤖 Model Configuration
- model_type: Type of model (default: "codellama")
- quantized_type: 🔧 Type of quantized model (optional)

#### 📚 Dataset Options
- dataset: 📚 Dataset to use ("humaneval" or "mbpp", default: "mbpp")
- mini: 🔍 Use mini version of dataset (flag)

#### 🎯 Attack Parameters
- attack_method: Type of attack ("synonym", "random upper", "translate-and-back")
- replacement_prob: Probability of replacement (default: 0.15)
- max_synonyms: Maximum number of synonyms (default: 3)
- input_type: Type of input (default: "prompt")
- seed: Random seed for reproducibility

#### 📦 Quantization Parameters
- quant_method: Static quantization method ("bnb", "gptq", "awq")
- quant_bits: Number of bits for quantization (4 or 8)
- quant_type: Quantization type for 4-bit static quantization ("nf4", "nf4_2", "nf4_3")
- quantize_embeddings: Whether to quantize embeddings (for dynamic quantization)

#### ⚙️ Generation Parameters
- num_return_sequences: Number of responses to generate (default: 1)
- max_length: Maximum generation length (default: 512)
- temperature: Temperature for sampling (default: 0.7)
- top_p: Top-p for sampling (default: 0.95)
- num_beams: Number of beams for beam search (default: 10)
- use_beam_search: Whether to use beam search (default: False)

## 📝 Examples

### 1. 🔰 Basic Usage:
```bash
# Attack original LLMs
adversarial-codegen attack \
    --model_path /path/to/model \
    --save_prompts /path/to/save/prompts \
    --save_results /path/to/save/results
```

### 2. 🚀 Advanced usage with custom parameters:
```bash
# Attack LLMs with a specific adversarial attack method (synonym) and generation method (temperature sampling).
adversarial-codegen attack \
    --model_path /path/to/model \
    --dataset mbpp \
    --attack_method synonym \
    --replacement_prob 0.2 \
    --max_synonyms 5 \
    --temperature 0.8 \
    --top_p 0.9 \
    --num_beams 5 \
    --seed 42 \
    --save_prompts /path/to/save/prompts \
    --save_results /path/to/save/results
```

### 3. 🔧 Using Static Quantization:
```bash
# Attack LLMs with static quant (4-bit quant achieved by bnb)
adversarial-codegen attack \
    --model_path /path/to/model \
    --quantized_type static \
    --quant_method bnb \
    --quant_bits 4 \
    --quant_type nf4 \
    --save_prompts /path/to/save/prompts \
    --save_results /path/to/save/results
```

### 4. 🔄 Using Dynamic Quantization:
```bash
# Attack LLMs with 8-bit quant
adversarial-codegen attack \
    --model_path /path/to/model \
    --quantized_type dynamic \
    --quant_bits 8 \
    --quantize_embeddings True \ # Generally don't quantize embedding layer
    --save_prompts /path/to/save/prompts \
    --save_results /path/to/save/results
```

## 📤 Output
The tool generates two types of outputs:

1. 📝 Prompts: Saved to the directory specified by --save_prompts
- Original prompts
- Adversarially modified prompts


2. 📊 Results: Saved to the directory specified by --save_results
- Model responses to original prompts
- Model responses to adversarial prompts
- Performance metrics and analysis (Now only include pass rate, visual statistics will come soon!)


## 👥 Contributing
We welcome contributions! Please feel free to submit a Pull Request.
For questions or suggestions, please contact:

- 📧 Email: <a href="mailto:fangsen1996@gmail.com">fangsen1996@gmail.com</a>/<a href="mailto:sfang9@ncsu.edu">sfang9@ncsu.edu</a>
- 💬 Open an issue
- 🔀 Submit a PR

## 🙏 Acknowledgments
This project builds upon and is inspired by several excellent works in the field:

- 🤗 HuggingFace Transformers - For transformer models and utilities
- 📚 MBPP Dataset - For evaluation datasets
- 🧪 HumanEval - For evaluation protocols and datasets
- ⚡ PEFT - For efficient model fine-tuning methods
- 🔍 EvalPlus - For enhanced evaluation methods

Special thanks to all these projects that made our work possible.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.



