import json
import os
from typing import Any, Dict, List, Optional

from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm import tqdm

from adversarial_codegen.attacks.char_attack import CharacterCaseAttack
from adversarial_codegen.attacks.synonym_attack import SynonymAttack
from adversarial_codegen.attacks.translation_attack import TranslationAttack
from adversarial_codegen.framework.base_attack import BaseAttack
from adversarial_codegen.models.base_model import BaseModel
from adversarial_codegen.utils.evaluation import evaluator


class AttackFramework:
    def __init__(self, 
                 model: BaseModel,
                 attack_method: str = "synonym",
                 attack_config: Dict[str, Any] = None,
                 dataset: str = "humaneval",
                 mini: bool = False):
        """
        Initialize attack framework.
        
        Args:
            model: Model to attack
            attack_method: Type of attack to use
            attack_config: Attack configuration
            dataset: Dataset to use ("humaneval" or "mbpp")
            mini: Whether to use mini version of dataset
        """
        self.model = model
        self.attack_method = attack_method
        self.attack_config = attack_config
        self.attacker = self._initialize_attacker()
        self.dataset = dataset.lower()
        self.mini = mini
        
        # Load appropriate dataset
        if self.dataset == "humaneval":
            self.problems = get_human_eval_plus(mini=mini)
            self.attack_config['input_type'] = 'code'
        elif self.dataset == "mbpp":
            self.problems = get_mbpp_plus(mini=mini)
            self.attack_config['input_type'] = 'prompt'
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Choose 'humaneval' or 'mbpp'")
    
    def _initialize_attacker(self) -> BaseAttack:
        """Initialize the appropriate attack method."""
        if self.attack_method == "synonym":
            print("Using synonym attack")
            return SynonymAttack(config=self.attack_config)
        elif self.attack_method == "char":
            print("Using character case attack")
            return CharacterCaseAttack(config=self.attack_config)
        elif self.attack_method == "translate":
            print("Using translation attack")
            return TranslationAttack(config=self.attack_config)
        else:
            raise ValueError(f"Unknown attack method: {self.attack_method}")
    
    def run_attack(self, sample_indices: Optional[List[int]] = None, save_prompts: str = None, save_results: str = None):
        """
        Run attack pipeline on selected problems.
        
        Args:
            sample_indices: Optional list of problem indices to attack.
                            If None, all problems will be used.
            save_prompts: Optional path to save prompts.
            save_results: Optional path to save results.
        
        Returns:
            Dictionary containing attack results and evaluation metrics
        """
        
        # Track generations
        original_generations = []
        adversarial_generations = []
        
        # Process each problem
        problems_to_attack = (
            list(self.problems.items()) if sample_indices is None 
            else [(k, v) for i, (k, v) in enumerate(self.problems.items()) 
                  if i in sample_indices]
        )

        for task_id, problem in tqdm(problems_to_attack):
            # Get appropriate prompt for dataset type
            # prompt = self._get_problem_prompt(problem)
            prompt = problem["prompt"]
            
            # Generate original output
            original_output = self.model.generate(prompt)
            
            adversarial_prompt = self.attacker.generate_adversarial_example(prompt)
            adversarial_output = self.model.generate(adversarial_prompt)
            
            # Format for evaluation
            original_generations.append({
                "task_id": task_id,
                "solution": original_output,
                "prompt": prompt,
            })
            
            adversarial_generations.append({
                "task_id": task_id,
                "solution": adversarial_output,
                "prompt": adversarial_prompt,
            })

        if save_prompts:
            if not os.path.exists(save_prompts):
                os.makedirs(save_prompts)
            save_adv_prompt = os.path.join(save_prompts, "adversarial_prompts.jsonl")
            save_ori_prompt = os.path.join(save_prompts, "original_prompts.jsonl")
            with open(save_adv_prompt, 'w') as f:
                for adv in adversarial_generations:
                    f.write(json.dumps(adv) + '\n')
            with open(save_ori_prompt, 'w') as f:
                for ori in original_generations:
                    f.write(json.dumps(ori) + '\n')
        
        original_results = evaluator(self.dataset, original_generations)
        adversarial_results = evaluator(self.dataset, adversarial_generations)
        
        if save_results:
            if not os.path.exists(save_results):
                os.makedirs(save_results)
            save_adv_results = os.path.join(save_results, "adversarial_results.json")
            save_ori_results = os.path.join(save_results, "original_results.json")
            with open(save_adv_results, 'w') as f:
                json.dump(adversarial_results, f)
            with open(save_ori_results, 'w') as f:
                json.dump(original_results, f)
        
        return original_results, adversarial_results


if __name__ == "__main__":
    from adversarial_codegen.models import DynamicQuantizedModel, Models

    # model = Models.load("codellama", model_path="/home/sfang9/workshop/llms/original_llms/deepseek-coder-6.7b-base")
    model = DynamicQuantizedModel(model_path="/home/sfang9/workshop/llms/original_llms/deepseek-coder-6.7b-base", quantized_path="/home/sfang9/workshop/llms/compressed_llms/deepseek_ai_deepseek_coder_6.7b_base_w8a8_cali500_None.pt")
    attack_config = {
        "replacement_probability": 0.15,
        "max_synonyms": 3,
        "input_type": "prompt",
        "seed": 42
    }
    attack_framework = AttackFramework(model=model, attack_method="synonym", attack_config=attack_config, dataset="mbpp")
    save_path = "aisec/adversarial-attack-nlp/pre_results/deepseek-coder-67b-compressed"
    _, _ = attack_framework.run_attack(save_prompts=save_path, save_results=save_path)
