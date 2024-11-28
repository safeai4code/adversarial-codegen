from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer"""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate code from prompt"""
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate code for multiple prompts"""
        pass