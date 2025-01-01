from typing import Dict, List, Type

from .base_model import BaseModel


class ModelRegistry:
    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls: Type[BaseModel]):
            cls._models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get_model(cls, name: str) -> Type[BaseModel]:
        if name not in cls._models:
            raise ValueError(f"Model {name} not found in registry")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._models.keys())
