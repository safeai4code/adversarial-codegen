import random
from typing import Any, Dict, List, Optional

from ..framework.base_attack import BaseAttack


class NaturalNoiseAttack(BaseAttack):
    """Adds natural programming noise like comments, whitespace, and variable renaming."""
    
    def validate_config(self) -> None:
        required_keys = ['noise_types', 'noise_probability']
        if not all(key in self.config for key in required_keys):
            raise ValueError(f"Config must contain: {required_keys}")
            
        if not 0 <= self.config['noise_probability'] <= 1:
            raise ValueError("noise_probability must be between 0 and 1")
    
    def generate_adversarial_example(
            self,
            input_code: str,
            target_label: Optional[Any] = None
        ) -> str:
        modified_code = input_code
        
        if 'comments' in self.config['noise_types']:
            modified_code = self._add_random_comments(modified_code)
            
        if 'whitespace' in self.config['noise_types']:
            modified_code = self._modify_whitespace(modified_code)
            
        if 'variable_rename' in self.config['noise_types']:
            modified_code = self._rename_variables(modified_code)
            
        return modified_code
    
    def _add_random_comments(self, code: str) -> str:
        # Implementation for adding random but natural-looking comments
        comments = [
            "# TODO: Optimize this later",
            "# FIXME: Consider edge cases",
            "# Note: This is a temporary solution"
        ]
        lines = code.split('\n')
        for i in range(len(lines)):
            if random.random() < self.config['noise_probability']:
                lines[i] = f"{lines[i]} {random.choice(comments)}"
        return '\n'.join(lines)
    
    def _modify_whitespace(self, code: str) -> str:
        # Implementation for modifying whitespace while preserving functionality
        lines = code.split('\n')
        for i in range(len(lines)):
            if random.random() < self.config['noise_probability']:
                lines[i] = "    " + lines[i]
        return '\n'.join(lines)
    
    def _rename_variables(self, code: str) -> str:
        # Basic implementation - would need AST parsing for robustness
        # This is a simplified version for demonstration
        common_vars = ['i', 'j', 'x', 'y', 'temp', 'data']
        new_names = ['var_' + str(i) for i in range(len(common_vars))]
        
        modified_code = code
        for old_name, new_name in zip(common_vars, new_names):
            if random.random() < self.config['noise_probability']:
                modified_code = modified_code.replace(f" {old_name} ", f" {new_name} ")
        
        return modified_code
    
    def _is_valid_python(self, code: str) -> bool:
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
