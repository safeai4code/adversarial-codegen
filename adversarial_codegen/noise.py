# Standard library imports
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import bitsandbytes as bnb
import fire
import numpy as np
import torch
import torch.nn as nn
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from adversarial_codegen.utils import extract_functions


@dataclass
class ModelConfig:
    model_path: str
    quantization: Optional[str] = None  # '4bit', '8bit', or None
    device_map: str = 'auto'


class NoiseConfig:
    def __init__(self, noise_type: str = "gaussian", noise_level: float = 1e-3):
        self.noise_type = noise_type
        self.noise_level = noise_level
        self._validate()
    
    def _validate(self):
        valid_noise_types = ["uniform", "gaussian"]
        if self.noise_type not in valid_noise_types:
            raise ValueError(f"Noise type must be one of {valid_noise_types}")
        if self.noise_level < 0:
            raise ValueError("Noise level must be positive")


class LLMEvaluator:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_model()
        
    def _setup_model(self):
        """Setup model with or without quantization"""
        # Setup quantization configuration if specified
        if self.model_config.quantization is None:
            # Load original model without quantization
            print("Loading original model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_path,
                device_map=self.model_config.device_map
            )
        else:
            # Setup quantization configuration
            if self.model_config.quantization == '4bit':
                print("Loading quantized model with 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float32,
                )
            elif self.model_config.quantization == '8bit':
                print("Loading quantized model with 8-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                raise ValueError(f"Unsupported quantization type: {self.model_config.quantization}")
            
            # Load quantized model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_path,
                quantization_config=quantization_config,
                device_map=self.model_config.device_map,
                torch_dtype=torch.float32 if self.model_config.quantization == '4bit' else torch.float16,
            )
        
        # Load tokenizer (same for both original and quantized models)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path)
        
    def add_noise(self, noise_config: NoiseConfig):
        """Add noise to model parameters"""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    if noise_config.noise_type == 'gaussian':
                        noise = torch.randn_like(param).to(param.device) * noise_config.noise_level
                    else:  # uniform
                        noise = (torch.rand_like(param).to(param.device) * 2 - 1) * noise_config.noise_level
                    param.add_(noise)
    
    def generate(self, prompt: str, generation_config: Optional[GenerationConfig] = None) -> str:
        """Generate text from prompt"""
        if generation_config is None:
            generation_config = GenerationConfig(
                max_length=500,
                do_sample=False,
                num_return_sequences=1,
            )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class LLMEvaluationCLI:
    """Command line interface for LLM evaluation with noise analysis"""
    
    @staticmethod
    def _prepare_output_file(output_path: str):
        """Prepare the output file and its directory.
        
        Args:
            output_path: Path where results will be saved
            
        Returns:
            Path object of the prepared output file
        """
        output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear existing file if it exists
        if output_path.exists():
            output_path.unlink()
            
        return output_path
    
    @staticmethod
    def evaluate_single_model(
        model_path: str,
        output_path: str = "results.jsonl",
        quantization: Optional[str] = None,
        noise_type: str = "uniform",
        noise_level: float = 1e-3,
        device_map: str = "auto",
        max_length: int = 500,
        num_return_sequences: int = 1,
        do_sample: bool = False
    ):
        """
        Evaluate a single model with specified configurations.
        
        Args:
            model_path: Path to the model
            output_path: Path to save results
            quantization: Quantization type ('4bit', '8bit', or None)
            noise_type: Type of noise to add ('uniform' or 'gaussian')
            noise_level: Level of noise to add
            device_map: Device mapping strategy
            max_length: Maximum length for generation
            num_return_sequences: Number of sequences to return
            do_sample: Whether to use sampling for generation
        """
        # Prepare output file and directory
        output_path = LLMEvaluationCLI._prepare_output_file(output_path)
        
        # Load prompts
        mbpp_problems = get_mbpp_plus()
        problems_to_eval = list(mbpp_problems.items())

        # Setup configurations
        model_config = ModelConfig(model_path, quantization, device_map)
        noise_config = NoiseConfig(noise_type, noise_level)
        generation_config = GenerationConfig(
            max_length=max_length,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
        )
        
        # Initialize evaluator
        noised_evaluator = LLMEvaluator(model_config)
        noised_evaluator.add_noise(noise_config)
        
        # Process each prompt
        results = []
        for task_id, problem in tqdm(problems_to_eval):
            
            prompt = problem["prompt"]
            
            # Generate with noise
            noised_generation = noised_evaluator.generate(prompt, generation_config)
            gen_solution = extract_functions(noised_generation[len(prompt):].lstrip())
            
            result = {
                "task_id": task_id,
                "solution": gen_solution,
            }
            
            results.append(result)
            
            # Write result to file
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
        
        return f"Evaluation completed. Results saved to {output_path}"
    
    @staticmethod
    def evaluate_multiple_models(
        model_paths: List[str],
        prompt_file: str,
        output_path: str = "results.jsonl",
        quantizations: Optional[List[str]] = None,
        noise_types: List[str] = ["uniform"],
        noise_levels: List[float] = [1e-4],
        device_map: str = "auto",
        max_length: int = 500,
        do_sample: bool = False
    ):
        """
        Evaluate multiple models with different configurations.
        
        Args:
            model_paths: List of paths to models
            prompt_file: Path to file containing prompts (one per line)
            output_path: Path to save results
            quantizations: List of quantization types for each model
            noise_types: List of noise types to test
            noise_levels: List of noise levels to test
            device_map: Device mapping strategy
            max_length: Maximum length for generation
            do_sample: Whether to use sampling for generation
        """
        # Prepare output file and directory
        output_path = LLMEvaluationCLI._prepare_output_file(output_path)
        
        if quantizations is None:
            quantizations = [None] * len(model_paths)
        
        for model_path, quantization in zip(model_paths, quantizations):
            for noise_type in noise_types:
                for noise_level in noise_levels:
                    print(f"Evaluating model: {model_path} with {noise_type} noise at level {noise_level}")
                    LLMEvaluationCLI.evaluate_single_model(
                        model_path=model_path,
                        prompt_file=prompt_file,
                        output_path=output_path,
                        quantization=quantization,
                        noise_type=noise_type,
                        noise_level=noise_level,
                        device_map=device_map,
                        max_length=max_length,
                        do_sample=do_sample
                    )
        
        return f"Evaluation completed for all models. Results saved to {output_path}"


def main():
    fire.Fire(LLMEvaluationCLI)


if __name__ == "__main__":
    main()
