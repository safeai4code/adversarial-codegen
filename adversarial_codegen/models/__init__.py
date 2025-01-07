from .base_model import BaseModel
from .model_implementations import (
    CodeGenModel,
    CodeLLaMAModel,
    DeepSeekModel,
    DynamicQuantizedModel,
    InCoderModel,
    MagicCoderModel,
    StarCoderModel,
    StaticQuantizedModel,
)
from .model_loader import Models
from .model_registry import ModelRegistry


# Register default models
@ModelRegistry.register("codellama")
class CodeLLaMAModel(CodeLLaMAModel): pass


@ModelRegistry.register("starcoder")
class StarCoderModel(StarCoderModel): pass


@ModelRegistry.register("codegen")
class CodeGenModel(CodeGenModel): pass


@ModelRegistry.register("deepseek")
class DeepSeekModel(DeepSeekModel): pass


@ModelRegistry.register("incoder")
class InCoderModel(InCoderModel): pass


@ModelRegistry.register("magicoder")
class MagicCoderModel(MagicCoderModel): pass


@ModelRegistry.register("dynamic")
class DynamicQuantizedModel(DynamicQuantizedModel): pass


@ModelRegistry.register("static")
class StaticQuantizedModel(StaticQuantizedModel): pass
