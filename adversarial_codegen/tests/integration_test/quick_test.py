from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional

import fire
import torch

from adversarial_codegen.models import Models
from adversarial_codegen.run import AdversarialCodeGen, AttackConfig
from adversarial_codegen.tests.testing.framework.test_attack_framework import (
    TestAttackFramework,
)
from adversarial_codegen.utils import visualizer


class TestAdversarialCodeGen(AdversarialCodeGen):
    def attack(
        self,
        model_path: str,
        model_type: str = "codellama",
        quantized_type: Optional[str] = None,
        dataset: str = "mbpp",
        attack_method: str = "synonym",
        save_prompts: Optional[str] = None,
        save_results: Optional[str] = None,
        visualization: bool = False,
        # Attack parameters
        replacement_prob: float = 0.15,
        max_synonyms: int = 3,
        char_change_probability: float = 0.5,
        max_char_changes: int = 15,
        translation_model: str = "facebook/mbart-large-50-many-to-many-mmt",
        attack_model: str = "gpt-4o",
        attack_type: str = "paraphrase",
        adv_tempature: float = 0.7,
        adv_max_tokens: int = 150,
        api_path: str = "",
        input_type: str = None,
        seed: Optional[int] = None,
        # Quantization parameters
        quant_method: Literal["bnb", "gptq", "awq"] = "bnb",
        quant_bits: Literal[4, 8] = 8,
        quant_type: Literal["nf4", "fp4"] = "nf4",
        quantize_embeddings: bool = False,
        # Generation parameters
        num_return_sequences: int = 1,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_beams: int = 10,
        use_beam_search: bool = False
    ):
        """
        Extended version of attack function with test parameter.
        
        Args:
            All original parameters from parent class
        """
        # Create configurations
        attack_config = AttackConfig(
            replacement_probability=replacement_prob,
            max_synonyms=max_synonyms,
            char_change_probability=char_change_probability,
            max_char_changes=max_char_changes,
            translation_model=translation_model,
            attack_model=attack_model,
            attack_type=attack_type,
            adv_tempature=adv_tempature,
            adv_max_tokens=adv_max_tokens,
            api_path=api_path,
            input_type=input_type,
            seed=seed
        )

        # Set up quantization and generation parameters
        quant_params = {
            "method": quant_method,
            "bits": quant_bits,
            "quant_type": quant_type,
            "quantize_embeddings": quantize_embeddings
        }
        
        gen_params = {
            "num_return_sequences": num_return_sequences,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "use_beam_search": use_beam_search
        }

        # Create model configuration
        model_config = self._create_model_config(
            model_type=model_type,
            quantized_type=quantized_type,
            quant_params=quant_params,
            gen_params=gen_params
        )

        # Initialize model with configurations
        model = Models.load(
            model_type if quantized_type is None else quantized_type,
            model_path=model_path,
            **model_config
        )

        # Initialize framework with test parameter
        framework = TestAttackFramework(
            model=model,
            attack_method=attack_method,
            attack_config=asdict(attack_config),
            dataset=dataset,
            mini=False
        )

        # Run attack
        original_results, adversarial_results = framework.run_attack(
            save_prompts=save_prompts,
            save_results=save_results
        )

        # Visualize the results
        if visualization:
            visualizer(original_results, adversarial_results, model_path.rsplit('/', 1)[-1], save_results)


def main():
    fire.Fire(TestAdversarialCodeGen)

if __name__ == "__main__":
    main()
