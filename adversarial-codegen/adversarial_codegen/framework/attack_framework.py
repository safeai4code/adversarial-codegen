from typing import Dict, Any, List, Optional
from ..models.base_model import BaseModel
from ..framework.base_attack import BaseAttack
from ..attacks.synonym_attack import SynonymAttack

class AttackFramework:
    def __init__(self, 
                 model: BaseModel,
                 attack_method: str,
                 attack_config: Dict[str, Any]):
        self.model = model
        self.attack_method = attack_method
        self.attack_config = attack_config
        self.attacker = self._initialize_attacker()
    
    def _initialize_attacker(self) -> BaseAttack:
        if self.attack_method == "synonym":
            return SynonymAttack(config=self.attack_config)
        raise ValueError(f"Unknown attack method: {self.attack_method}")
    
    def run_attack(self, 
                   inputs: List[Dict[str, str]],
                   evaluation_metric: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run attack pipeline: generate adversarial input -> get model output -> evaluate.
        
        Args:
            inputs: List of input dicts with 'type' and 'content' keys
            evaluation_metric: Optional metric for evaluating attack success
            
        Returns:
            List of results for each input
        """
        results = []
        
        for input_item in inputs:
            # Get original model output
            original_output = self.model.generate(input_item['content'])
            
            # Generate adversarial input
            if input_item['type'] == 'prompt':
                adv_input = self.attacker.attack_prompt(input_item['content'])
            elif input_item['type'] == 'code':
                adv_input = self.attacker.attack_code_comments(input_item['content'])
            else:
                raise ValueError(f"Unknown input type: {input_item['type']}")
            
            # Get model output for adversarial input
            adversarial_output = self.model.generate(adv_input)
            
            result = {
                'original_input': input_item['content'],
                'adversarial_input': adv_input,
                'original_output': original_output,
                'adversarial_output': adversarial_output
            }
            
            # verify attack success
            if evaluation_metric:
                # Placeholder for evaluation.py integration
                # TODO: Implement evaluation logic
                # result['evaluation'] = {}
                pass
        
        return results