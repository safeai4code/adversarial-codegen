import random
import re
from typing import Any, Dict, List, Optional

from adversarial_codegen.framework.base_attack import BaseAttack


class CharacterCaseAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validate_config()
        # Initialize random seed if provided
        self.seed = config.get('seed')
        if self.seed is not None:
            random.seed(self.seed)

    def validate_config(self) -> None:
        """Validate the configuration parameters."""
        required = ['char_change_probability', 'max_char_changes', 'input_type', 'seed']
        if not all(key in self.config for key in required):
            raise ValueError(f"Config must contain: {required}")
        if not 0 <= self.config['char_change_probability'] <= 1:
            raise ValueError("char_change_probability must be between 0 and 1")
        if not isinstance(self.config['max_char_changes'], int) or self.config['max_char_changes'] < 0:
            raise ValueError("max_char_changes must be a non-negative integer")
        if 'seed' in self.config and not isinstance(self.config['seed'], (int, type(None))):
            raise ValueError("seed must be an integer or None")

    def generate_adversarial_example(self, input_text: str, target_label: Optional[Any] = None) -> str:
        """Generate an adversarial example by randomly changing character cases."""
        if self.seed is not None:
            random.seed(self.seed)

        if self.config['input_type'] == 'prompt':
            input_text_lines = input_text.splitlines()
            assert len(input_text_lines) == 4, "Unknown prompt format"
            # Attack only the prompt natural language text
            attack_line = self._attack_text(input_text_lines[1])
            return '\n'.join([input_text_lines[0], attack_line, input_text_lines[2], input_text_lines[3]])
        elif self.config['input_type'] == 'code':
            return self._attack_code_comments(input_text)
        raise ValueError(f"Unknown input type: {self.config['input_type']}")

    def _attack_text(self, text: str) -> str:
        """Apply character case transformation to text with a limit on total changes."""
        # Get positions of all alphabetic characters
        alpha_positions = [i for i, char in enumerate(text) if char.isalpha()]
        
        if not alpha_positions:
            return text
            
        # Calculate how many characters to change
        max_possible_changes = min(
            len(alpha_positions),  # Can't change more than available alpha chars
            self.config['max_char_changes']  # Can't exceed max_char_changes
        )
        
        # Determine number of changes based on probability and max limit
        probable_changes = sum(
            1 for _ in range(len(alpha_positions))
            if random.random() < self.config['char_change_probability']
        )
        num_changes = min(probable_changes, max_possible_changes)
        
        # Randomly select positions to change
        positions_to_change = random.sample(alpha_positions, num_changes)
        
        # Apply changes
        result = list(text)
        for pos in positions_to_change:
            result[pos] = result[pos].upper()
        
        return ''.join(result)

    def _attack_code_comments(self, code: str) -> str:
        """Apply character case transformation to docstrings while preserving code."""
        # Pattern to find triple-quoted strings (both single and double quotes)
        docstring_pattern = r'(\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\")'
        
        def replace_docstring(match):
            """Helper function to process each docstring match."""
            docstring = match.group(0)
            quote_type = docstring[:3]  # Get the type of quotes used (''' or """)
            # Extract the content between the quotes
            content = docstring[3:-3]
            # Apply character case transformation to the content
            modified_content = self._attack_text(content)
            # Reconstruct the docstring with the same quote type
            return f"{quote_type}{modified_content}{quote_type}"
        
        # Replace all docstrings in the code
        modified_code = re.sub(docstring_pattern, replace_docstring, code)
        return modified_code


if __name__ == "__main__":
    # Example usage
    attack = CharacterCaseAttack(config={
        'char_change_probability': 0.5,
        'max_char_changes': 5,  # Maximum 5 characters can be changed
        'input_type': 'prompt',
        'seed': 42
    })
    attack.validate_config()
    
    # Test prompt
    prompt = "\"\"\"\nWrite a function to find the shared elements from the given two lists.\n" \
             "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n\"\"\"\n"
    
    # Generate multiple attacks with the same seed
    attack_prompt_1 = attack.generate_adversarial_example(prompt)
    attack_prompt_2 = attack.generate_adversarial_example(prompt)
    attack_prompt_3 = attack.generate_adversarial_example(prompt)
    
    # Verify reproducibility with seed
    assert attack_prompt_1 == attack_prompt_2
    assert attack_prompt_2 == attack_prompt_3
    
    print("Original prompt line:")
    print(prompt.splitlines()[1])
    print("\nTransformed prompt line:")
    print(attack_prompt_1.splitlines()[1])
