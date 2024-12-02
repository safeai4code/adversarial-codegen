from typing import List, Dict, Any, Optional
from ..models.base_model import BaseModel
from .base_attack import BaseAttack

class AttackFramework:
    """Orchestrates adversarial attacks against code generation models."""
    
    def __init__(self, 
                 model: BaseModel,
                 attacks: List[BaseAttack],
                 config: Dict[str, Any]):
        self.model = model
        self.attacks = attacks
        self.config = config
        
    def run_attack(self,
                  input_code: str,
                  target_label: Optional[Any] = None) -> Dict[str, Any]:
        """Run all configured attacks on the input code.
        
        Args:
            input_code: Original code to attack
            target_label: Optional target label for targeted attacks
            
        Returns:
            Dictionary containing attack results and metrics
        """
        results = {}
        original_output = self.model.generate(input_code)
        
        for attack in self.attacks:
            try:
                adversarial_code = attack.generate_adversarial_example(
                    input_code, target_label)
                adversarial_output = self.model.generate(adversarial_code)
                
                success = attack.attack_success_criteria(
                    original_output, adversarial_output)
                
                results[attack.__class__.__name__] = {
                    'success': success,
                    'original_code': input_code,
                    'adversarial_code': adversarial_code,
                    'original_output': original_output,
                    'adversarial_output': adversarial_output
                }
            except Exception as e:
                results[attack.__class__.__name__] = {
                    'success': False,
                    'error': str(e)
                }
                
        return results