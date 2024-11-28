from ..attacks.natural_noise_attack import NaturalNoiseAttack
from ..attacks.structural_attack import StructuralAttack
from ..attacks.semantic_attack import SemanticAttack


class AttackFramework:
    def __init__(self):
        self.supported_attacks = {
            "natural_noise": NaturalNoiseAttack,
            "structural": StructuralAttack,
            "semantic": SemanticAttack
        }

    def run_attack(self, model, dataset, attack_type, params=None):
        if attack_type not in self.supported_attacks:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        
        attack = self.supported_attacks[attack_type](params)
        results = []
        
        for example in dataset:
            attacked_input = attack.generate(example)
            original_output = model.generate(example)
            attacked_output = model.generate(attacked_input)
            
            results.append({
                "original_input": example,
                "attacked_input": attacked_input,
                "original_output": original_output,
                "attacked_output": attacked_output
            })
        
        return results

    def generate_report(self, results):
        # Implement report generation logic
        pass