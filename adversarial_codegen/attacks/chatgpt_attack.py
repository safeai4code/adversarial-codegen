import os
import random
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from adversarial_codegen.framework.base_attack import BaseAttack


class AttackType(Enum):
    """Attack types supported by the ChatGPTAttack class."""
    PARAPHRASE = "paraphrase"
    CONSTRAINT_CHANGE = "constraint_change"
    SCOPE_EXPANSION = "scope_expansion"
    SEMANTIC_PRESERVE = "semantic_preserve"


class ChatGPTAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validate_config()
        self.api_key = self._load_api_key()
        self.client = OpenAI(api_key=self.api_key)
        self.model = config.get('attack_model', 'gpt-4o')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 150)
        self.attack_type = config.get('attack_type', AttackType.PARAPHRASE)

    def _load_api_key(self) -> str:
        """Load OpenAI API key from .env_adversarial file."""
        env_file = Path(self.config['api_path'])
        if not env_file.exists():
            raise FileNotFoundError(
                "'.env_adversarial' file not found. Please create it with your OpenAI API key."
            )
        
        with open(env_file, 'r') as f:
            api_key = f.read().strip()
            
        if not api_key:
            raise ValueError("API key cannot be empty in .env_adversarial file")
            
        return api_key

    def validate_config(self) -> None:
        """Validate the configuration parameters."""
        required = ['input_type', 'api_path']
        if not all(key in self.config for key in required):
            raise ValueError(f"Config must contain: {required}")
        if not isinstance(self.config['api_path'], str):
            print(self.config)
            raise ValueError("openai_api_key must be a string")
        if self.config['input_type'] not in ['prompt', 'code']:
            raise ValueError("input_type must be either 'prompt' or 'code'")
        
        # Validate attack_type if provided
        if 'attack_type' in self.config:
            self.config['attack_type'] = AttackType(self.config['attack_type'])
            if not isinstance(self.config['attack_type'], AttackType):
                raise ValueError("attack_type must be an instance of AttackType enum")
        
    def _generate_attack_prompt(self, text: str, attack_type: AttackType) -> str:
        """Generate the prompt based on the attack type."""
        if attack_type == AttackType.PARAPHRASE:
            return (
                "Rephrase the following programming task description while keeping the exact same meaning. "
                "The new version should be clear and natural but use different wording. "
                "Provide only the rephrased version without any additional text or explanations.\n\n"
                f"Original: {text}"
            )
        elif attack_type == AttackType.CONSTRAINT_CHANGE:
            return (
                "Modify the following programming task by slightly adjusting its constraints "
                "while maintaining the core objective. Make the changes subtle but meaningful. "
                "Provide only the modified version without explanations.\n\n"
                f"Original: {text}"
            )
        elif attack_type == AttackType.SCOPE_EXPANSION:
            return (
                "Expand the scope of the following programming task slightly. "
                "Add one or two additional requirements that naturally extend the original task. "
                "Provide only the expanded version without explanations.\n\n"
                f"Original: {text}"
            )
        elif attack_type == AttackType.SEMANTIC_PRESERVE:
            return (
                "Transform the following programming task by preserving its semantic meaning "
                "but expressing it in a completely different way. Use alternative concepts "
                "or approaches that achieve the same goal. "
                "Provide only the transformed version without explanations.\n\n"
                f"Original: {text}"
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

    def _get_chatgpt_response(self, prompt: str, attack_type: AttackType) -> str:
        """Get response from ChatGPT API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant that generates {attack_type.value} "
                                    f"variations of programming tasks."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Error getting response from ChatGPT: {str(e)}")

    def generate_adversarial_example(
        self, 
        input_text: str,
        attack_type: Optional[AttackType] = None
    ) -> str:
        """Generate an adversarial example using the specified attack type."""
        # Use provided attack_type or fall back to the one from config
        current_attack = attack_type if attack_type is not None else self.attack_type
        
        if self.config['input_type'] == 'prompt':
            input_text_lines = input_text.splitlines()
            assert len(input_text_lines) == 4, "Unknown prompt format"
            
            # Get the natural language description (line 1)
            original_description = input_text_lines[1]
            
            # Generate attack prompt and get ChatGPT response
            attack_prompt = self._generate_attack_prompt(original_description, current_attack)
            modified_description = self._get_chatgpt_response(attack_prompt, current_attack)
            
            # Return the modified prompt with the same structure
            return '\n'.join([
                input_text_lines[0],
                modified_description,
                input_text_lines[2],
                input_text_lines[3]
            ])
        elif self.config['input_type'] == 'code':
            raise NotImplementedError("Code modification not implemented yet")
        raise ValueError(f"Unknown input type: {self.config['input_type']}")


if __name__ == "__main__":
    # Example usage
    attack = ChatGPTAttack(config={
        'input_type': 'prompt',
        'attack_model': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'max_tokens': 150,
        'api_path': "/home/sfang9/workshop/project_test/openai/openai-key",
        'attack_type': 'paraphrase'
    })
    
    # Test prompt
    prompt = '\"\"\"\nWrite a function to find the shared elements from the given two lists.\n' \
             'assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n\"\"\"\n'
    
    # Try different attack types
    for attack_type in AttackType:
        print(f"\nTrying {attack_type.value} attack:")
        try:
            adversarial_prompt = attack.generate_adversarial_example(prompt, attack_type)
            print("Original prompt line:")
            print(prompt.splitlines()[1])
            print("\nTransformed prompt line:")
            print(adversarial_prompt.splitlines()[1])
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Try customized attack type
    try:
        adversarial_prompt = attack.generate_adversarial_example(prompt)
        print("\noriginal prompt:")
        print(prompt.splitlines()[1])
        print("\nCuatomized attack prompt:")
        print(adversarial_prompt.splitlines()[1])
    except Exception as e:
        print(f"Error: {str(e)}")
