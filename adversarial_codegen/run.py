from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional

import fire
import torch

from adversarial_codegen.framework.attack_framework import AttackFramework
from adversarial_codegen.models import Models
from adversarial_codegen.utils import visualizer


@dataclass
class AttackConfig:
    """Configuration for attack parameters"""
    # synonym replacement attack parameters
    replacement_probability: float = 0.15
    max_synonyms: int = 3
    # character case attack parameters
    char_change_probability: float = 0.5,
    max_char_changes: int = 5,
    # translation attack parameters
    translation_model: str = "facebook/mbart-large-50-many-to-many-mmt",
    # LLM-based attack parameters
    attack_model: str = "gpt-4o",
    attack_type: str = "paraphrase",
    adv_tempature: float = 0.7,
    adv_max_tokens: int = 150,
    api_path: str = "",
    # General attack parameters
    input_type: str = "prompt"
    seed: Optional[int] = None


class AdversarialCodeGen:
    @staticmethod
    def _create_model_config(
        model_type: str,
        quantized_type: Optional[str],
        quant_params: dict,
        gen_params: dict
    ) -> dict:
        """Create model configuration including quantization and generation settings"""
        model_config = {}
        
        # Add generation config if provided
        if gen_params:
            model_config["generation_config"] = {
                "num_return_sequences": gen_params.get("num_return_sequences", 1),
                "max_length": gen_params.get("max_length", 512),
                "temperature": gen_params.get("temperature", 0.7),
                "top_p": gen_params.get("top_p", 0.95),
                "num_beams": gen_params.get("num_beams", 10),
                "use_beam_search": gen_params.get("use_beam_search", False)
            }

        # Add quantization config if using quantization
        if quantized_type == "static":
            model_config["quant_config"] = {
                "method": quant_params.get("method", "bnb"),
                "bits": quant_params.get("bits", 8),
                "compute_dtype": torch.float16,
                "quant_type": quant_params.get("quant_type", "nf4"),
                "dataset": quant_params.get("dataset", "c4")
            }
        elif quantized_type == "dynamic":
            model_config["quant_config"] = {
                "bits": quant_params.get("bits", 8),
                "quantize_embeddings": quant_params.get("quantize_embeddings", False)
            }

        return model_config

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
        # mini: bool = False,
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
        Run adversarial attack on code with detailed parameter control.
        
        Args:
            # Base parameters
            model_path: Path to the model,
            model_type: Type of model (codellama, starcoder, etc.)
                # TODO: Use decoder-only, encoder-only, and encoeer-decoder to specify model type in the future.
            quantized_type: Type of quantization (None, "static", or "dynamic").
            dataset: Dataset to use, choices=["mbpp", "humaneval"].
                # TODO: Add support for mini version
            attack_method: Type of attack, choices=["synonym", "char", "translate", "llm_attack"].
            save_prompts: Path to save prompts.
            save_results: Path to save results. Required if visualization is True.
            
            # Attack parameters
            replacement_prob: Probability of replacement.
            max_synonyms: Maximum number of synonyms.
            char_change_probability: Probability of changing character case.
            max_char_changes: Maximum number of character changes.
            translation_model: Translation model for translation attack. Cho
            attack_model: Model for LLM-based attack. Now only support ChatGPT.
            attack_type: Type of attack for LLM-based attack. Choices=["paraphrase", "constraint_change",
                        "scope_expansion", "semantic_preserve"].
            adv_tempature: Temperature for LLM-based attack.
            adv_max_tokens: Maximum tokens for LLM-based attack.
            input_type: Type of input, decided by the dataset.
            seed: Random seed for reproducibility.
            
            # Quantization parameters
            quant_method: Static quantization method. Choices=["bnb", "gptq", "awq"].
            quant_bits: Number of bits for quantization.
                Note: Only 4 and 8 are supported for static quantization and 8 for dynamic quantization.
            quant_type: Quantization type for 4-bit static quantization. Choices=["nf4", "nf4_2", "nf4_3"].
            quantize_embeddings: Whether to quantize embeddings (for dynamic).
            
            # Generation parameters
            num_return_sequences: Number of responses to generate. Note: if set to 1, greedy decoding is used.
            max_length: Maximum generation length.
            temperature: Temperature for sampling.
            top_p: Top-p for sampling, generally used with temperature.
            num_beams: Number of beams for beam search.
                Note: Only used if use_beam_search is True and should be equal or greater than num_return_sequences.
            use_beam_search: Whether to use beam search.
        """
        # Check if save_results is provided when visualization is enabled
        if visualization and save_results is None:
            raise ValueError("save_results must be provided when visualization is enabled")
        
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

        # Initialize framework
        framework = AttackFramework(
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
    fire.Fire(AdversarialCodeGen)

if __name__ == "__main__":
    main()
