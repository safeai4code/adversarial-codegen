# utils/evaluation.py
from typing import Dict, Any, List
from evalplus.data import get_human_eval_plus, get_mbpp_plus
import tempfile
import json
import os

class EvalPlusEvaluator:
    def __init__(self):
        self.human_eval_plus = get_human_eval_plus()
        self.mbpp_plus = get_mbpp_plus()

    def evaluate_code(self, task_id: str, code: str) -> Dict[str, Any]:
        tmp_dir = tempfile.mkdtemp()
        completion_path = os.path.join(tmp_dir, "completion.jsonl")
        
        # Format completion for EvalPlus
        completion = {
            "task_id": task_id,
            "completion": code,
            "generation_params": {"temperature": 0}
        }
        
        with open(completion_path, "w") as f:
            json.dump(completion, f)
            f.write("\n")
            
        # Run evaluation
        result = test_solutions(
            completion_path,
            problem_file=self.human_eval_plus,
            base_problem_file=self.human_eval,
            k=[1]
        )
        
        return {
            "pass_rate": result["pass@1"],
            "base_pass_rate": result["base_pass@1"],
            "failures": result.get("failures", [])
        }

def evaluate_code_generations(original_outputs: List[str], 
                            adversarial_outputs: List[str], 
                            task_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Evaluate original and adversarial code generations using EvalPlus.
    
    Args:
        original_outputs: List of original code generations
        adversarial_outputs: List of adversarial code generations
        task_ids: List of HumanEval task IDs corresponding to each output
        
    Returns:
        List of evaluation results comparing original vs adversarial outputs
    """
    evaluator = EvalPlusEvaluator()
    results = []
    
    for orig, adv, task_id in zip(original_outputs, adversarial_outputs, task_ids):
        orig_result = evaluator.evaluate_code(task_id, orig)
        adv_result = evaluator.evaluate_code(task_id, adv)
        
        results.append({
            "task_id": task_id,
            "original": {
                "code": orig,
                "evaluation": orig_result
            },
            "adversarial": {
                "code": adv,
                "evaluation": adv_result
            },
            "attack_success": orig_result["pass_rate"] > adv_result["pass_rate"]
        })
    
    return results