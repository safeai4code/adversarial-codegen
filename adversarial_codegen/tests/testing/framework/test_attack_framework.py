import random
from dataclasses import asdict
from typing import Any, Dict, List, OrderedDict

from evalplus.data import get_human_eval_plus, get_mbpp_plus

from adversarial_codegen.framework.attack_framework import AttackFramework
from adversarial_codegen.framework.base_attack import BaseAttack
from adversarial_codegen.models.base_model import BaseModel


class TestAttackFramework(AttackFramework):
    def __init__(self, 
                 model: BaseModel,
                 attack_method: str = "synonym",
                 attack_config: Dict[str, Any] = None,
                 dataset: str = "humaneval",
                 mini: bool = False,
                 testset_size: int = 5,
                 random_seed: int = 42):
        """
        Initialize test attack framework.
        
        Args:
            testset_size: Number of problems to select from dataset in test mode
            random_seed: Seed for random selection
            Rest is the same as AttackFramework
        """
        self.testset_size = testset_size
        self.random_seed = random_seed
        
        # Initialize parent class
        super().__init__(model=model,
                        attack_method=attack_method,
                        attack_config=attack_config,
                        dataset=dataset,
                        mini=False)
        
        self.problems = self._get_fixed_subset()
    
    def _get_fixed_subset(self) -> Dict[str, Any]:
        """
        Get a fixed subset of problems that will be consistent across runs.

        Returns:
            Dictionary containing the selected problems
        """
        if self.testset_size >= len(self.problems):
            return self.problems
        
        problems_list = list(self.problems.items())
        random.seed(self.random_seed)
        selected_problems = random.sample(problems_list, self.testset_size)
        return OrderedDict(selected_problems)
