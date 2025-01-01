from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAttack(ABC):
    """Base class for all adversarial attacks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """Validate attack-specific configuration."""
        pass
    
    @abstractmethod
    def generate_adversarial_example(
            self, 
            input_code: str,
            target_label: Optional[Any] = None
        ) -> str:
        """Generate an adversarial example from the input code.
        
        Args:
            input_code: Original code snippet
            target_label: Optional target label for targeted attacks
            
        Returns:
            Modified code with adversarial perturbations
        """
        pass
