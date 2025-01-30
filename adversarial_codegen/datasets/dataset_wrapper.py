import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from adversarial_codegen.attacks.chatgpt_attack import AttackType
from adversarial_codegen.framework.base_attack import BaseAttack


class AdversarialDatasetWrapper:
    """A wrapper class to generate and manage adversarial examples dataset."""
    
    def __init__(
        self,
        attack_model: BaseAttack,
    ):
        """
        Initialize the dataset wrapper.
        
        Args:
            attack_model: Instance of ChatGPTAttack to use for generating adversarial examples
        """
        self.attack_model = attack_model
        
        # Initialize directories
        self._initialize_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize cache indices for each attack type
        self.cache_indices = {}
        for attack_type in AttackType:
            cache_index_path = self.cache_dir / f"cache_index_{attack_type.value}.json"
            self.cache_indices[attack_type] = self._load_cache_index(cache_index_path)

    def _initialize_directories(self):
        """Initialize cache and log directories using environment variables."""
        # Get cache and log directories from environment variables or use defaults
        default_cache = str(Path.home() / ".adversarial_cache")
        default_logs = str(Path.home() / ".adversarial_logs")
        
        self.cache_dir = Path(os.getenv("ADVERSARIAL_CACHE_DIR", default_cache))
        self.log_dir = Path(os.getenv("ADVERSARIAL_LOG_DIR", default_logs))
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.log_dir / f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _load_cache_index(self, cache_index_path: Path) -> Dict:
        """Load a cache index from disk or create a new one."""
        if cache_index_path.exists():
            with open(cache_index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self, attack_type: AttackType):
        """Save a cache index to disk."""
        cache_index_path = self.cache_dir / f"cache_index_{attack_type.value}.json"
        with open(cache_index_path, 'w') as f:
            json.dump(self.cache_indices[attack_type], f, indent=2)
            
    def _get_cache_key(self, original_prompt: str) -> str:
        """Generate a unique cache key for a prompt."""
        import hashlib
        return hashlib.md5(original_prompt.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str, attack_type: AttackType) -> Optional[str]:
        """Load an adversarial example from cache if it exists."""
        if attack_type not in self.cache_indices or cache_key not in self.cache_indices[attack_type]:
            return None
            
        cache_file = Path(self.cache_indices[attack_type][cache_key]['prompt_file'])
        if not cache_file.exists():
            return None
            
        with open(cache_file, 'r') as f:
            return f.read()
    
    def _save_to_cache(
        self,
        cache_key: str,
        attack_type: AttackType,
        adversarial_prompt: str,
        metadata: Dict
    ):
        """Save an adversarial example and its metadata to cache."""
        cache_file = self.cache_dir / f"{attack_type.value}_{cache_key}.txt"
        metadata_file = self.cache_dir / f"{attack_type.value}_{cache_key}_metadata.json"
        
        # Save adversarial example
        with open(cache_file, 'w') as f:
            f.write(adversarial_prompt)
            
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Update cache index
        if attack_type not in self.cache_indices:
            self.cache_indices[attack_type] = {}
            
        self.cache_indices[attack_type][cache_key] = {
            'prompt_file': str(cache_file),
            'metadata_file': str(metadata_file)
        }
        self._save_cache_index(attack_type)
    
    def verify_single_generation(self, cache_key: str, attack_type: AttackType) -> bool:
        """
        Verify that a prompt was only generated once for a specific attack type.
        
        Args:
            cache_key: The cache key to verify
            attack_type: The attack type to verify
            
        Returns:
            bool: True if the prompt was only generated once
        """
        if attack_type not in self.cache_indices or cache_key not in self.cache_indices[attack_type]:
            return True
            
        metadata_file = Path(self.cache_indices[attack_type][cache_key]['metadata_file'])
        if not metadata_file.exists():
            return False
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        return metadata.get('generation_count', 0) == 1
    
    def generate_dataset(
        self,
        prompts: List[str],
        attack_type: Optional[AttackType] = None,
        force_regenerate: bool = False
    ) -> List[Tuple[str, str]]:
        """
        Generate adversarial examples for a list of prompts.
        
        Args:
            prompts: List of original prompts
            attack_type: Type of attack to use (defaults to model's configured type)
            force_regenerate: If True, regenerate examples even if cached
            
        Returns:
            List of tuples (original_prompt, adversarial_prompt)
        """
        results = []
        current_attack = attack_type if attack_type is not None else self.attack_model.attack_type
        
        for prompt in tqdm(prompts, desc=f"Generating {current_attack.value} adversarial examples"):
            cache_key = self._get_cache_key(prompt)
            
            # Verify single generation
            if not force_regenerate and not self.verify_single_generation(cache_key, current_attack):
                logging.warning(
                    f"Prompt {cache_key} has been generated multiple times "
                    f"for {current_attack.value} attack!"
                )
                continue
                
            # Check cache
            if not force_regenerate:
                cached_example = self._load_from_cache(cache_key, current_attack)
                if cached_example is not None:
                    results.append((prompt, cached_example))
                    continue
            
            try:
                logging.info(
                    f"Generating {current_attack.value} adversarial example "
                    f"for prompt {cache_key}"
                )
                
                # Generate adversarial example
                start_time = time.time()
                adversarial_prompt = self.attack_model.generate_adversarial_example(
                    prompt, current_attack
                )
                generation_time = time.time() - start_time
                
                # Prepare metadata
                metadata = {
                    'original_prompt': prompt,
                    'attack_type': current_attack.value,
                    'timestamp': datetime.now().isoformat(),
                    'model': self.attack_model.model,
                    'temperature': self.attack_model.temperature,
                    'generation_time': generation_time,
                    'generation_count': 1
                }
                
                # Save to cache
                self._save_to_cache(cache_key, current_attack, adversarial_prompt, metadata)
                
                logging.info(
                    f"Successfully generated and cached {current_attack.value} "
                    f"example {cache_key}"
                )
                results.append((prompt, adversarial_prompt))
                
            except Exception as e:
                logging.error(f"Error generating adversarial example: {str(e)}")
                continue
                
        return results
    
    def load_dataset(self, attack_type: Optional[AttackType] = None) -> List[Tuple[str, str]]:
        """
        Load the complete dataset of original and adversarial examples from cache.
        
        Args:
            attack_type: Specific attack type to load (loads all types if None)
            
        Returns:
            List of tuples (original_prompt, adversarial_prompt)
        """
        results = []
        
        attack_types = [attack_type] if attack_type else AttackType
        
        for current_attack in attack_types:
            if current_attack not in self.cache_indices:
                continue
                
            for cache_key, files in self.cache_indices[current_attack].items():
                cache_file = Path(files['prompt_file'])
                metadata_file = Path(files['metadata_file'])
                
                if cache_file.exists() and metadata_file.exists():
                    with open(cache_file, 'r') as f:
                        adversarial_prompt = f.read()
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        
                    original_prompt = metadata['original_prompt']
                    results.append((original_prompt, adversarial_prompt))
                
        return results
    
    def clear_cache(self, attack_type: Optional[AttackType] = None):
        """
        Clear cached adversarial examples.
        
        Args:
            attack_type: Specific attack type to clear (clears all if None)
        """
        attack_types = [attack_type] if attack_type else AttackType
        
        for current_attack in attack_types:
            # Remove cache files for the attack type
            for pattern in [
                f"{current_attack.value}_*.txt",
                f"{current_attack.value}_*.json",
                f"cache_index_{current_attack.value}.json"
            ]:
                for file in self.cache_dir.glob(pattern):
                    file.unlink()
            
            # Reset cache index
            if current_attack in self.cache_indices:
                self.cache_indices[current_attack] = {}


if __name__ == "__main__":
    # Example usage showing how to set environment variables
    import os

    from adversarial_codegen.attacks.chatgpt_attack import AttackType, ChatGPTAttack

    # Set environment variables (in practice, these would be set outside the script)
    os.environ["ADVERSARIAL_CACHE_DIR"] = str(Path.home() / "research" / "adversarial_cache")
    os.environ["ADVERSARIAL_LOG_DIR"] = str(Path.home() / "research" / "adversarial_logs")

    # Initialize attack model
    attack = ChatGPTAttack(config={
        'input_type': 'prompt',
        'model': 'gpt-4o',
        'temperature': 0.7,
        'max_tokens': 150,
        'api_path': "/home/sfang9/workshop/project_test/openai/openai-key",
        'attack_type': AttackType.PARAPHRASE
    })
    
    # Initialize dataset wrapper
    wrapper = AdversarialDatasetWrapper(
        attack_model=attack,
    )
    
    # Test prompts
    test_prompts = [
        '''"""
Write a function to find the shared elements from the given two lists.
assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
"""
''',
        '''"""
Implement a function that calculates the factorial of a given number.
assert factorial(5) == 120
"""
'''
    ]
    
    # Generate dataset with different attack types
    for attack_type in AttackType:
        print(f"\nGenerating dataset with {attack_type.value} attack:")
        results = wrapper.generate_dataset(test_prompts, attack_type)
        
        print(f"\nResults for {attack_type.value}:")
        for original, adversarial in results:
            print("\nOriginal:", original.splitlines()[1])
            print("Adversarial:", adversarial.splitlines()[1])
            print("-" * 50)
            
    # Load complete dataset
    all_results = wrapper.load_dataset()
    print(f"\nTotal examples in dataset: {len(all_results)}")
