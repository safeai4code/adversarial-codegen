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

## 🚀 Supported Models
### Original LLMs / Compressed LLMs

<!-- Replace {path_to_logo} with actual paths after adding logos to your repo's assets folder -->

- <img src="./assets/logos/codellama.png" width="20"> CodeLLaMA  <!-- Download from Meta's official repo -->
- <img src="./assets/logos/starcoder.png" width="20"> StarCoder  <!-- From BigCode/HuggingFace -->
- <img src="./assets/logos/codegen.png" width="20"> CodeGen     <!-- From Salesforce -->
- <img src="./assets/logos/deepseek.png" width="20"> DeepSeek   <!-- From DeepSeek official website -->
- <img src="./assets/logos/incoder.png" width="20"> InCoder     <!-- From Meta/Facebook -->
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
