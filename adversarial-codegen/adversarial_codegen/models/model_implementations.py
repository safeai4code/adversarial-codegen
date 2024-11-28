from typing import Optional, List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .base_model import BaseModel


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