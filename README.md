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

## 🚀 Supported Models (Original LLMs / Compressed LLMs)

- <img src="./assets/llama_logo.jpg" width="20"> [CodeLLaMA](https://github.com/meta-llama/codellama)  <!-- Download from Meta's official repo -->
- <img src="./assets/starcoder_logo.png" width="20"> [StarCoder](https://github.com/bigcode-project/starcoder)  <!-- From BigCode/HuggingFace -->
- <img src="./assets/codegen_logo.png" width="20"> [CodeGen](https://github.com/salesforce/CodeGen)     <!-- From Salesforce -->
- <img src="./assets/deepseek_logo.png" width="20"> [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder-V2)   <!-- From DeepSeek official website -->
- <img src="./assets/incoder_logo.png" width="20"> [InCoder](https://github.com/dpfried/incoder)     <!-- From Meta/Facebook -->
- 🎩 [Magicoder](https://github.com/ise-uiuc/magicoder)

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

<!--- 
## ⚙️ Installation
```bash
git clone https://github.com/yourusername/adversarial-codegen
cd adversarial-codegen
pip install -r requirements.txt
