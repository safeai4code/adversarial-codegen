from typing import Dict, Any, List, Optional
import os
import tempfile
from evalplus.data import (
    get_human_eval_plus, 
    get_mbpp_plus,
    write_jsonl
)
from adversarial_codegen.models.base_model import BaseModel
from adversarial_codegen.framework.base_attack import BaseAttack 
from adversarial_codegen.attacks.synonym_attack import SynonymAttack
from adversarial_codegen.utils.evaluation import evaluate_adversarial_attack

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
        elif self.dataset == "mbpp":
            self.problems = get_mbpp_plus(mini=mini)
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Choose 'humaneval' or 'mbpp'")
    
    def _initialize_attacker(self) -> BaseAttack:
        """Initialize the appropriate attack method."""
        if self.attack_method == "synonym":
            return SynonymAttack(config=self.attack_config)
        raise ValueError(f"Unknown attack method: {self.attack_method}")
    
    # def _get_problem_prompt(self, problem: Dict[str, Any]) -> str:
    #     """Get the appropriate prompt based on dataset type."""
    #     if self.dataset == "humaneval":
    #         return problem["prompt"]
    #     else:  # mbpp
    #         # MBPP format includes test cases in the prompt
    #         prompt = problem["prompt"] + "\n\n"
    #         # Add test cases as part of the prompt
    #         for i, (input_case, output_case) in enumerate(
    #             zip(problem["test_inputs"], problem["test_outputs"]), 1
    #         ):
    #             prompt += f"# Test Case {i}:\n"
    #             prompt += f"assert {problem['entry_point']}({input_case}) == {output_case}\n"
    #         return prompt
    
    def run_attack(self, sample_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run attack pipeline on selected problems.
        
        Args:
            sample_indices: Optional list of problem indices to attack.
                          If None, all problems will be used.
        
        Returns:
            Dictionary containing attack results and evaluation metrics
        """
        # Set up temp directories for outputs
        temp_dir = tempfile.mkdtemp()
        original_path = os.path.join(temp_dir, "original_generations.jsonl")
        adversarial_path = os.path.join(temp_dir, "adversarial_generations.jsonl")
        
        # Track generations
        original_generations = []
        adversarial_generations = []
        
        # Process each problem
        problems_to_attack = (
            list(self.problems.items()) if sample_indices is None 
            else [(k, v) for i, (k, v) in enumerate(self.problems.items()) 
                  if i in sample_indices]
        )
        breakpoint()
        
        for task_id, problem in problems_to_attack:
            # Get appropriate prompt for dataset type
            # prompt = self._get_problem_prompt(problem)
            prompt = problem["prompt"]
            
            # Generate original output
            original_output = self.model.generate(prompt)
            
            # Generate adversarial prompt by attacking docstring/comments
            if self.dataset == "humaneval":
                self.attack_config['input_type'] = 'code'
            elif self.dataset == "mbpp":
                self.attack_config['input_type'] = 'prompt'
            else:
                raise ValueError(f"Unknown dataset: {self.dataset}")
            
            adversarial_prompt = self.attacker.generate_adversarial_example(prompt)
            adversarial_output = self.model.generate(adversarial_prompt)
            
            # Format for evaluation
            original_generations.append({
                "task_id": task_id,
                "completion": original_output,
                "prompt": prompt,
                "entry_point": problem["entry_point"]  # Required for MBPP
            })
            
            adversarial_generations.append({
                "task_id": task_id,
                "completion": adversarial_output,
                "prompt": adversarial_prompt,
                "entry_point": problem["entry_point"]
            })
        
        # Write generations to files
        write_jsonl(original_path, original_generations)
        write_jsonl(adversarial_path, adversarial_generations)
        
        # Run evaluation
        results = evaluate_adversarial_attack(
            original_generations=original_generations,
            adversarial_generations=adversarial_generations,
            dataset=self.dataset,
            mini=self.mini
        )
        
        # Add attack details to results
        results["attack_details"] = {
            "method": self.attack_method,
            "config": self.attack_config,
            "dataset": self.dataset,
            "num_samples": len(problems_to_attack)
        }
        
        # Add generation details
        results["generations"] = {
            task_id: {
                "original": {
                    "prompt": orig["prompt"],
                    "completion": orig["completion"],
                    "entry_point": orig["entry_point"]
                },
                "adversarial": {
                    "prompt": adv["prompt"],
                    "completion": adv["completion"],
                    "entry_point": adv["entry_point"]
                }
            }
            for task_id, orig, adv in zip(
                [g["task_id"] for g in original_generations],
                original_generations,
                adversarial_generations
            )
        }
        
        return results


if __name__ == "__main__":
    from adversarial_codegen.models import Models
    model = Models.load("codellama", model_path="/home/sfang9/workshop/llms/original_llms/Llama-3.2-1B")
    attack_config = {
        "replacement_probability": 0.5,
        "max_synonyms": 5
    }
    attack_framework = AttackFramework(model=model, attack_method="synonym", attack_config=attack_config, dataset="mbpp")