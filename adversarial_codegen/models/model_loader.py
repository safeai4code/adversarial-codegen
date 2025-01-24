from typing import Optional, Type

from .base_model import BaseModel
from .model_registry import ModelRegistry

# from .model_implementations import *


class Models:
    DEFAULT_MODELS = {
        "codellama": "codellama/CodeLlama-7b-hf",
        "starcoder": "bigcode/starcoder",
        "codegen": "Salesforce/codegen-350M-mono",
        "deepseek": "deepseek-ai/deepseek-coder-1.3b-base",
        "incoder": "facebook/incoder-1B",
        "magicoder": "ise-uiuc/Magicoder-CL-7B"
    }

    @staticmethod
    def load(model_name: str, model_path: Optional[str] = None, **kwargs) -> BaseModel:
        """
        Load a model by name or custom path
        """
        if model_path is None:
            if model_name in Models.DEFAULT_MODELS:
                model_path = Models.DEFAULT_MODELS[model_name]
            else:
                raise ValueError(f"Model {model_name} not found in default models")

        model_cls = ModelRegistry.get_model(model_name)
        return model_cls(model_path=model_path, **kwargs)

    @staticmethod
    def register_model(name: str, model_class: Type[BaseModel]):
        """
        Register a new model type
        """
        ModelRegistry.register(name)(model_class)
