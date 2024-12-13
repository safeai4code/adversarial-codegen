import os
from typing import Optional, List, Union
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

from .base_model import BaseModel


@dataclass
class GenerationStrategy:
    """Configuration for different generation strategies"""
    num_return_sequences: int = 1
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    num_beams: int = 10
    use_beam_search: bool = False


class CodeLLaMAModel(BaseModel):
    def __init__(self, model_path: str = "codellama/CodeLlama-7b-hf", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def _get_generation_config(self, strategy: GenerationStrategy) -> GenerationConfig:
        """Create generation configuration based on strategy"""
        if strategy.num_return_sequences == 1:
            # Use greedy decoding for single sequence
            print("Using greedy decoding")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=1,
                do_sample=False,
            )
        elif strategy.use_beam_search:
            # Use beam search for multiple sequences
            print("Using beam search")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=strategy.num_return_sequences,
                num_beams=strategy.num_beams,
                do_sample=False,
            )
        else:
            # Use temperature sampling for multiple sequences
            print("Using temperature sampling")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=strategy.num_return_sequences,
                temperature=strategy.temperature,
                top_p=strategy.top_p,
                do_sample=True,
            )
    
    def _extract_completion(self, full_text: str, prompt: str) -> str:
        """Extract only the completion part from the generated text"""
        if full_text.startswith(prompt):
            return full_text[len(prompt):].lstrip()
        return full_text

    def generate(
        self, 
        prompt: str,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate completion(s) for a given prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Generation parameters to override defaults in GenerationStrategy
                     (num_return_sequences, max_length, temperature, top_p, num_beams, use_beam_search)
        
        Returns:
            Single string if num_return_sequences=1, otherwise list of strings
        """
        # Start with default strategy and update with any provided kwargs
        strategy = GenerationStrategy()
        if kwargs:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(kwargs)
            strategy = GenerationStrategy(**strategy_dict)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_config = self._get_generation_config(strategy)
        
        outputs = self.model.generate(
            inputs.input_ids,
            generation_config=generation_config
        )

        # Decode all sequences
        decoded_outputs = [
            self._extract_completion(
                self.tokenizer.decode(output, skip_special_tokens=True),
                prompt
            )
            for output in outputs
        ]

        # Return single string if num_return_sequences=1, otherwise list
        return decoded_outputs[0] if strategy.num_return_sequences == 1 else decoded_outputs

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class DynamicQuantizedModel(BaseModel):
    def __init__(self, model_path: str, quantized_path: str, **kwargs):
        """
        Initialize the dynamically quantized model.
        
        Args:
            model_path: Path to the original model's tokenizer (e.g., "codellama/CodeLlama-7b-hf")
            quantized_path: Path to the quantized model weights (.pt file)
            **kwargs: Additional arguments
        """
        super().__init__(model_path, **kwargs)
        self.quantized_path = quantized_path
        if not os.path.exists(self.quantized_path):
            raise ValueError(f"Quantized model file not found: {self.quantized_path}")
        self.load()

    def load(self) -> None:
        """Load the tokenizer and quantized model (CPU-only)"""
        # Load tokenizer from original model path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # Load quantized model on CPU
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.quantized_path,
        #     device_map="auto",  # Force CPU
        #     load_in_8bit=True,  # Enable 8-bit loading
        #     torch_dtype=torch.float16,  # Base dtype for non-quantized weights
        #     low_cpu_mem_usage=True
        # )
        self.model = torch.load(self.quantized_path, map_location="cpu")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()

    def _get_generation_config(self, strategy: GenerationStrategy) -> GenerationConfig:
        """Create generation configuration based on strategy"""
        if strategy.num_return_sequences == 1:
            print("Using greedy decoding")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=1,
                do_sample=False,
            )
        elif strategy.use_beam_search:
            print("Using beam search")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=strategy.num_return_sequences,
                num_beams=strategy.num_beams,
                do_sample=False,
            )
        else:
            print("Using temperature sampling")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=strategy.num_return_sequences,
                temperature=strategy.temperature,
                top_p=strategy.top_p,
                do_sample=True,
            )

    def _extract_completion(self, full_text: str, prompt: str) -> str:
        """Extract only the completion part from the generated text"""
        if full_text.startswith(prompt):
            return full_text[len(prompt):].lstrip()
        return full_text

    def generate(
        self, 
        prompt: str,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate completion(s) for a given prompt using CPU-only quantized model.
        
        Args:
            prompt: Input prompt text
            **kwargs: Generation parameters to override defaults in GenerationStrategy
                     (num_return_sequences, max_length, temperature, top_p, num_beams, use_beam_search)
        
        Returns:
            Single string if num_return_sequences=1, otherwise list of strings
        """
        # Start with default strategy and update with any provided kwargs
        strategy = GenerationStrategy()
        if kwargs:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(kwargs)
            strategy = GenerationStrategy(**strategy_dict)

        # Ensure inputs are on CPU
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_config = self._get_generation_config(strategy)

        # outputs = self.model.generate(
        #     inputs.input_ids,
        #     generation_config=generation_config
        # )
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                generation_config=generation_config
            )

        # Decode all sequences
        decoded_outputs = [
            self._extract_completion(
                self.tokenizer.decode(output, skip_special_tokens=True),
                prompt
            )
            for output in outputs
        ]

        # Return single string if num_return_sequences=1, otherwise list
        return decoded_outputs[0] if strategy.num_return_sequences == 1 else decoded_outputs

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate completions for multiple prompts"""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class StarCoderModel(BaseModel):
    def __init__(self, model_path: str = "bigcode/starcoder", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()
    
    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=kwargs.get('max_length', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class CodeGenModel(BaseModel):
    def __init__(self, model_path: str = "Salesforce/codegen-350M-mono", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=kwargs.get('max_length', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class DeepSeekModel(BaseModel):
    def __init__(self, model_path: str = "deepseek-ai/deepseek-coder-1.3b-base", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=kwargs.get('max_length', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class InCoderModel(BaseModel):
    def __init__(self, model_path: str = "facebook/incoder-1B", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=kwargs.get('max_length', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    

class MagicCoderModel(BaseModel):
    def __init__(self, model_path: str = "ise-uiuc/Magicoder-CL-7B", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=kwargs.get('max_length', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]