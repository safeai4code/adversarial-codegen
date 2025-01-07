import random
import re
from typing import Any, Dict, Optional

from transformers import pipeline

"""
This is a backup of the translation_attack.py for acheiving more general generation configurations.
"""


class TranslationAttack:
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validate_config()
        
        # Initialize random seed if provided
        self.seed = config.get('seed')
        if self.seed is not None:
            random.seed(self.seed)
            
        # Get generation parameters
        self.temperature = config.get('temperature', 0.7)  # Default temperature for sampling
        self.num_beams = config.get('num_beams', 5)  # Default number of beams
        self.do_sample = config.get('do_sample', True)  # Enable sampling by default
        
        # Initialize translation models
        self.model_name = config.get('model_name', 'facebook/mbart-large-50-many-to-many-mmt')
        self.device = config.get('device', 'cuda')
        
        # Initialize translators with generation parameters
        generation_kwargs = {
            'temperature': self.temperature,
            'num_beams': self.num_beams,
            'do_sample': self.do_sample,
            # Add top_k and top_p for nucleus sampling
            'top_k': config.get('top_k', 50),
            'top_p': config.get('top_p', 0.95),
            # Prevent repetition
            'no_repeat_ngram_size': 3,
            # For diverse outputs
            'num_beam_groups': 3 if self.num_beams > 3 else 1,
            'diversity_penalty': 0.5,
        }
        
        self.en_to_de = pipeline(
            "translation",
            model=self.model_name,
            src_lang="en_XX",
            tgt_lang="de_DE",
            device=self.device,
            **generation_kwargs
        )
        
        self.de_to_en = pipeline(
            "translation",
            model=self.model_name,
            src_lang="de_DE",
            tgt_lang="en_XX",
            device=self.device,
            **generation_kwargs
        )

    def validate_config(self) -> None:
        """Validate the configuration parameters."""
        required = ['input_type']
        if not all(key in self.config for key in required):
            raise ValueError(f"Config must contain: {required}")
        
        if 'seed' in self.config and not isinstance(self.config['seed'], (int, type(None))):
            raise ValueError("seed must be an integer or None")
            
        if 'model_name' in self.config and not isinstance(self.config['model_name'], str):
            raise ValueError("model_name must be a string")
            
        if 'temperature' in self.config and not (0 < self.config['temperature'] <= 2):
            raise ValueError("temperature must be between 0 and 2")

    def generate_adversarial_example(self, input_text: str, target_label: Optional[Any] = None) -> str:
        """Generate adversarial example through back translation."""
        if self.seed is not None:
            random.seed(self.seed)

        if self.config['input_type'] == 'prompt':
            input_text_lines = input_text.splitlines()
            assert len(input_text_lines) == 4, "Unknown prompt format"
            # Attack the prompt natural language text
            attack_line = self._attack_prompt(input_text_lines[1])
            return '\n'.join([input_text_lines[0], attack_line, input_text_lines[2], input_text_lines[3]])
        elif self.config['input_type'] == 'code':
            return self._attack_code_comments(input_text)
        raise ValueError(f"Unknown input type: {self.config['input_type']}")

    def _attack_prompt(self, prompt: str) -> str:
        """Apply back translation to natural language prompt."""
        # Translate English to German
        german = self.en_to_de(prompt)[0]['translation_text']
        
        # Translate German back to English
        back_translated = self.de_to_en(german)[0]['translation_text']
        
        return back_translated

    def _attack_code_comments(self, code: str) -> str:
        """Apply back translation to docstring comments while preserving code."""
        # Pattern to find triple-quoted strings (both single and double quotes)
        docstring_pattern = r'(\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\")'
        
        def replace_docstring(match):
            """Helper function to process each docstring match."""
            docstring = match.group(0)
            quote_type = docstring[:3]  # Get the type of quotes used (''' or """)
            # Extract the content between the quotes
            content = docstring[3:-3]
            # Apply back translation to the content
            modified_content = self._attack_prompt(content)
            # Reconstruct the docstring with the same quote type
            return f"{quote_type}{modified_content}{quote_type}"
        
        # Replace all docstrings in the code
        modified_code = re.sub(docstring_pattern, replace_docstring, code)
        return modified_code


if __name__ == "__main__":
    # Example usage with sampling parameters
    config = {
        'input_type': 'prompt',
        'seed': None,  # Allow randomness
        'model_name': 'facebook/mbart-large-50-many-to-many-mmt',
        'device': 'cuda',
        'temperature': 0.7,  # Higher values make output more diverse
        'num_beams': 5,
        'do_sample': True,  # Enable sampling
        'top_k': 50,  # Limit vocabulary for sampling
        'top_p': 0.95  # Nucleus sampling parameter
    }
    
    attack = TranslationAttack(config)
    
    prompt = '\"\"\"\nWrite a function to find the shared elements from the given two lists.\n' \
             'assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n\"\"\"\n'
    
    # Generate multiple examples to demonstrate variability
    for i in range(4):
        adversarial_prompt = attack.generate_adversarial_example(prompt)
        print(f"\nAdversarial Example {i+1}:")
        print(adversarial_prompt)
