import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Set, Union

import torch
from torch import dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig

from adversarial_codegen.utils.function_extractor import extract_functions

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


@dataclass
class QuantizationConfig:
    method: Literal["bnb", "gptq", "awq"] = "bnb"  # Quantization method
    bits: Literal[4, 8] = 8  # Bits for quantization
    compute_dtype: torch.dtype = torch.float16  # Computation dtype
    quant_type: str = "nf4"  # For bnb 4-bit: "nf4" or "fp4"
    dataset: str = "c4"  # For GPTQ calibration


@dataclass
class DynamicQuantizationConfig:
    """Configuration for dynamic quantization of Large Language Models.
    
    Attributes:
        bits: Currently only supports 8-bit for dynamic quantization
        dtype: Quantization dtype (only qint8 supported)
        qconfig_preset: Backend configuration ('fbgemm' for x86)
        quantize_embeddings: Whether to quantize embedding layers
        modules_to_quantize: Set of specific modules to quantize in transformer
    """
    bits: int = 8  # Fixed to 8 for dynamic quantization
    dtype: dtype = torch.qint8
    qconfig_preset: str = 'fbgemm'
    quantize_embeddings: bool = False  # Usually keep embeddings in full precision
    modules_to_quantize: Set[torch.nn.Module] = None

    def __post_init__(self):
        # Set default modules for transformer architecture
        if self.modules_to_quantize is None:
            self.modules_to_quantize = {
                torch.nn.Linear,  # For attention & feed-forward layers
                torch.nn.LayerNorm  # For layer normalization
            }
        
        self._validate()

    def _validate(self):
        """Validate the configuration parameters."""
        if self.bits != 8:
            raise ValueError("Dynamic quantization only supports 8-bit quantization")
        
        if self.dtype != torch.qint8:
            raise ValueError("Dynamic quantization only supports torch.qint8 dtype")
            
        if self.qconfig_preset != 'fbgemm':
            raise ValueError("LLM dynamic quantization only supports 'fbgemm' backend")

    def get_qconfig(self) -> torch.quantization.QConfig:
        """Get the PyTorch quantization config."""
        return torch.quantization.get_default_qconfig('fbgemm')

    def __str__(self) -> str:
        """String representation of the config."""
        return (
            f"DynamicQuantizationConfig(\n"
            f"  bits={self.bits},\n"
            f"  dtype={self.dtype},\n"
            f"  qconfig_preset='{self.qconfig_preset}',\n"
            f"  quantize_embeddings={self.quantize_embeddings},\n"
            f"  modules_to_quantize={self.modules_to_quantize}\n"
            f")"
        )


class CodeLLaMAModel(BaseModel):
    def __init__(self, model_path: str = "codellama/CodeLlama-7b-hf", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()
        self.gen_config = kwargs.get('generation_config')

    def load(self) -> None:
        print("Now loading original LLM model")
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
        
        output = full_text[len(prompt):].lstrip()
        gen_solution = extract_functions(output)

        if gen_solution is not None:
            return gen_solution
        else:
            return output

    def generate(
        self,
        prompt: str,
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
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_config = self._get_generation_config(strategy)

        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=self.tokenizer.pad_token_id
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
    def __init__(self, model_path: str = "codellama/CodeLlama-7b-hf", **kwargs):
        # Extract quantization config and generation config from kwargs
        try:
            self.quant_config = DynamicQuantizationConfig(**kwargs.get('quant_config'))
        except TypeError:
            raise ValueError("Quantization parameters not provided")
        self.gen_config = kwargs.get('generation_config')
        
        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        """Load tokenizer and dynamically quantized model"""
        print(f"Loading model for dynamic quantization ({self.quant_config.bits} bits)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model in FP32 on CPU for dynamic quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
        ).cpu()

        # Prepare modules to quantize based on config
        modules_to_quantize = self.quant_config.modules_to_quantize
        if not self.quant_config.quantize_embeddings:
            # Exclude embedding layers if specified
            print("Excluding embedding layers from quantization")
            exclude_modules = {torch.nn.Embedding}
            modules_to_quantize = {m for m in modules_to_quantize if m not in exclude_modules}

        # Apply dynamic quantization
        print("Applying dynamic quantization...")
        try:
            self.model = torch.quantization.quantize_dynamic(
                model,
                modules_to_quantize,
                dtype=self.quant_config.dtype
            )
            print("Dynamic quantization complete")
        except Exception as e:
            raise RuntimeError(f"Dynamic quantization failed: {str(e)}")

        print(self.get_memory_stats())

    def get_memory_stats(self) -> dict:
        """Get CPU memory statistics for dynamically quantized model"""
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "total_cpu_memory": memory_info.rss / 1024**3,  # GB
            "model_size": sum(p.nelement() * p.element_size() for p in self.model.parameters()) / 1024**3,  # GB
            "quantization_type": "dynamic",
            "bits": self.quant_config.bits,
            "device": "cpu"  # Dynamic quantization is CPU-only
        }

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
        
        output = full_text[len(prompt):].lstrip()
        gen_solution = extract_functions(output)

        if gen_solution is not None:
            return gen_solution
        else:
            return output

    def generate(
        self, 
        prompt: str,
    ) -> Union[str, List[str]]:
        """Generate completions using the quantized model on CPU"""
        # Reuse your existing generation code but ensure CPU operation
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        inputs = self.tokenizer(prompt, return_tensors="pt")  # Already on CPU
        generation_config = self._get_generation_config(strategy)
        
        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=self.tokenizer.pad_token_id
        )

        decoded_outputs = [
            self._extract_completion(
                self.tokenizer.decode(output, skip_special_tokens=True),
                prompt
            )
            for output in outputs
        ]

        return decoded_outputs[0] if strategy.num_return_sequences == 1 else decoded_outputs

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch generation - implemented as sequential for CPU operation"""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class StaticQuantizedModel(BaseModel):
    def __init__(self, model_path: str = "codellama/CodeLlama-7b-hf", **kwargs):
        # Extract quantization config and generation config from kwargs
        try:
            self.quant_config = QuantizationConfig(**kwargs.get('quant_config'))
        except TypeError:
            raise ValueError("Quantization parameters not provided")
        self.gen_config = kwargs.get('generation_config')
        super().__init__(model_path, **kwargs)

        # Initialize model with a static quantization configuration
        self.load()

    def load(self) -> None:
        """Load tokenizer and quantized model based on configuration"""

        assert self.quant_config.bits in [4, 8], "Bits must be 4 or 8"
        assert self.quant_config.quant_type in ["nf4", "fp4"], "Quant type must be 'nf4' or 'fp4'"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.quant_config.method == "bnb":
            print(f"Loading model with BitsAndBytes quantization ({self.quant_config.bits} bits)")
            self._load_bnb_model()
        elif self.quant_config.method == "gptq":
            self._load_gptq_model()
        elif self.quant_config.method == "awq":
            self._load_awq_model()
        else:
            raise ValueError(f"Unsupported quantization method: {self.quant_config.method}")

    def _load_bnb_model(self) -> None:
        """Load model with bitsandbytes quantization"""
        if self.quant_config.bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.quant_config.compute_dtype,
                bnb_4bit_quant_type=self.quant_config.quant_type
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            torch_dtype=self.quant_config.compute_dtype,
            device_map="auto"
        )

        print(self.get_memory_stats())

    def _load_gptq_model(self) -> None:
        """Load model with GPTQ quantization"""
        from transformers import GPTQConfig

        gptq_config = GPTQConfig(
            bits=self.quant_config.bits,
            dataset=self.quant_config.dataset,
            use_exllama=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=gptq_config,
            device_map="auto"
        )

    def _load_awq_model(self) -> None:
        """Load model with AWQ quantization"""
        if self.quant_config.bits != 8:
            raise ValueError("AWQ currently only supports 8-bit quantization")
            
        from awq import AutoAWQForCausalLM
        
        self.model = AutoAWQForCausalLM.from_pretrained(
            self.model_path,
            bits=self.quant_config.bits,
            device_map="auto"
        )

    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
        }

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
        
        output = full_text[len(prompt):].lstrip()
        gen_solution = extract_functions(output)

        if gen_solution is not None:
            return gen_solution
        else:
            return output

    def generate(
        self, 
        prompt: str,
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
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_config = self._get_generation_config(strategy)
        
        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=self.tokenizer.pad_token_id
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
