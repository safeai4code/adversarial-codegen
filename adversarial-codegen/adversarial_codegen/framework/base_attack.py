from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

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
    def generate_adversarial_example(self, 
                                   input_code: str,
                                   target_label: Optional[Any] = None) -> str:
        """Generate an adversarial example from the input code.
        
        Args:
            input_code: Original code snippet
            target_label: Optional target label for targeted attacks
            
        Returns:
            Modified code with adversarial perturbations
        """
        pass
    
    @abstractmethod
    def attack_success_criteria(self, 
                              original_output: Any,
                              adversarial_output: Any) -> bool:
        """Determine if the attack was successful.
        
        Args:
            original_output: Model output for original input
            adversarial_output: Model output for adversarial input
            
        Returns:
            True if attack succeeded, False otherwise
        """
        pass