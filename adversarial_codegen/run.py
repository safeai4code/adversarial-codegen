import fire
from adversarial_codegen.models import Models
from adversarial_codegen.framework.attack_framework import AttackFramework


class AdversarialCodeGen:
    def attack(self,
               model_path: str,
               model_type: str = "codellama",
               quantized_type: str = None,
               bits: int = None,
               static_quantized_method: str = None,
               dataset: str = "mbpp",
               attack_method: str = "synonym",
               save_prompts: str = "/path/to/save",
               save_results: str = "/path/to/save",
               replacement_prob: float = 0.15,
               max_synonyms: int = 3,
               input_type: str = "prompt",
               seed: int = None,
               mini: bool = False):
        """
        Run adversarial attack on code.
        
        Args:
            model_path: Path to the original model.
            model_type: Type of model (currently supports 'Causal LLMs').
            quantized_type: Type of quantized model (optional) and only used if quantized_path is provided. Choices are 'dynamic' and 'static'.
            dataset: Dataset to use ('humaneval' or 'mbpp'). Default is 'mbpp'.
            attack_method: Type of attack (currently supports 'synonym').
            save_dir: Directory to save results and prompts.
            replacement_prob: Probability of replacement for synonym attack.
            max_synonyms: Maximum number of synonyms to use.
            input_type: Type of input (currently supports 'prompt' and 'code').
            seed: Random seed.
            mini: Whether to use mini version of dataset.
        """
        # Configure attack
        attack_config = {
            "replacement_probability": replacement_prob,
            "max_synonyms": max_synonyms,
            "input_type": input_type,
            "seed": seed
        }

        # Initialize model
        if quantized_type == "dynamic":
            model = Models.load("dynamic", model_path=model_path)
        elif quantized_type == "static":
            if bits is None:
                print("Bits not provided, defaulting to 8 bits.")
            if static_quantized_method is None:
                print("Quantization method not provided, defaulting to 'bnb'.")
            
            # Think a way to better pass the quant_config to the model
            # TODO: Except for the attack config, we should also have a quantization config and a generation config
            # quant_config = {
            #     "bits": bits,
            #     "method": static_quantized_method
            # }

            model = Models.load("static", model_path=model_path)
        else:
            # Non-quantized model -> original LLMs
            model = Models.load("codellama", model_path=model_path)

        # Initialize framework
        framework = AttackFramework(
            model=model,
            attack_method=attack_method,
            attack_config=attack_config,
            dataset=dataset,
            mini=mini
        )

        # Run attack
        original_results, adversarial_results = framework.run_attack(
            save_prompts=save_prompts,
            save_results=save_results
        )

        #TODO: Next steps: Print the statistical results
        # return {"original": original_results, "adversarial": adversarial_results}

def main():
    fire.Fire(AdversarialCodeGen)

if __name__ == "__main__":
    main()