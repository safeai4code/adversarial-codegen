import random
import re
from typing import Any, Dict, List, Optional

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from adversarial_codegen.framework.base_attack import BaseAttack


class SynonymAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validate_config()
        self.stop_words = set(stopwords.words('english'))
        self.replaceable_pos = {'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'RB'}

        # Initialize random seed if provided
        self.seed = config.get('seed')
        if self.seed is not None:
            random.seed(self.seed)

    def validate_config(self) -> None:
        required = ['replacement_probability', 'max_synonyms', 'input_type', "seed"]
        if not all(key in self.config for key in required):
            raise ValueError(f"Config must contain: {required}")
        if not 0 <= self.config['replacement_probability'] <= 1:
            raise ValueError("replacement_probability must be between 0 and 1")
        if 'seed' in self.config and not isinstance(self.config['seed'], (int, type(None))):
            raise ValueError("seed must be an integer or None")

    def generate_adversarial_example(self, input_text: str, target_label: Optional[Any] = None) -> str:
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
        """Apply synonym replacement to natural language prompt."""
        tokens = word_tokenize(prompt)
        pos_tags = pos_tag(tokens)
        
        modified_tokens = []
        for word, pos in pos_tags:
            if (pos[:2] in self.replaceable_pos and 
                word.lower() not in self.stop_words and
                random.random() < self.config['replacement_probability']):
                synonym = self._find_synonym(word, pos)
                modified_tokens.append(synonym if synonym else word)
            else:
                modified_tokens.append(word)
        
        return self._reconstruct_text(modified_tokens)
    
    def _attack_code_comments(self, code: str) -> str:
        """Apply synonym replacement to docstring comments while preserving code."""
        # Pattern to find triple-quoted strings (both single and double quotes)
        docstring_pattern = r'(\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\")'
        
        def replace_docstring(match):
            """Helper function to process each docstring match."""
            docstring = match.group(0)
            quote_type = docstring[:3]  # Get the type of quotes used (''' or """)
            # Extract the content between the quotes
            content = docstring[3:-3]
            # Apply synonym replacement to the content
            modified_content = self.attack_prompt(content)
            # Reconstruct the docstring with the same quote type
            return f"{quote_type}{modified_content}{quote_type}"
        
        # Replace all docstrings in the code
        modified_code = re.sub(docstring_pattern, replace_docstring, code)
        return modified_code
    
    def _find_synonym(self, word: str, pos: str) -> str:
        """Find a synonym for a word based on its part of speech."""
        synsets = wn.synsets(word)
        
        if not synsets:
            return word
        
        # Convert POS tag to WordNet POS
        wn_pos = self._get_wordnet_pos(pos)
        if wn_pos:
            synsets = [s for s in synsets if s.pos() == wn_pos]
        
        if not synsets:
            return word
        
        # Get all lemmas from synsets
        lemmas = []
        for synset in synsets:
            lemmas.extend(synset.lemmas())
        
        # Get unique lemma names (excluding the original word)
        synonyms = list(set(lemma.name() for lemma in lemmas 
                          if lemma.name().lower() != word.lower()))
        
        if not synonyms:
            return word
        
        # Select a random synonym
        num_synonyms = min(len(synonyms), self.config['max_synonyms'])
        return random.choice(synonyms[:num_synonyms])
    
    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """Convert Penn Treebank POS tags to WordNet POS tags."""
        tag_map = {
            'JJ': wn.ADJ,
            'JJR': wn.ADJ,
            'JJS': wn.ADJ,
            'NN': wn.NOUN,
            'NNS': wn.NOUN,
            'NNP': wn.NOUN,
            'NNPS': wn.NOUN,
            'RB': wn.ADV,
            'RBR': wn.ADV,
            'RBS': wn.ADV,
            'VB': wn.VERB,
            'VBD': wn.VERB,
            'VBG': wn.VERB,
            'VBN': wn.VERB,
            'VBP': wn.VERB,
            'VBZ': wn.VERB
        }
        return tag_map.get(treebank_tag[:2])
    
    def _reconstruct_text(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens while handling punctuation properly."""
        text = ' '.join(tokens)
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?)])', r'\1', text)
        text = re.sub(r'(\()\s+', r'\1', text)
        return text


if __name__ == "__main__":
    attack = SynonymAttack(config={
        'replacement_probability': 0.5,
        'max_synonyms': 3,
        'input_type': 'prompt',
        'seed': 42
    })
    attack.validate_config()
    prompt = "\"\"\"\nWrite a function to find the shared elements from the given two lists.\n" \
             "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n\"\"\"\n"
    attach_prompt_1 = attack.generate_adversarial_example(prompt)
    attach_prompt_2 = attack.generate_adversarial_example(prompt)
    attach_prompt_3 = attack.generate_adversarial_example(prompt)
    attach_prompt_4 = attack.generate_adversarial_example(prompt)
 
    assert attach_prompt_1 == attach_prompt_2 # we set seed to 42, so the result should be the same
    assert attach_prompt_2 == attach_prompt_3
    assert attach_prompt_3 == attach_prompt_4
    print(attach_prompt_1, attach_prompt_2, attach_prompt_3, attach_prompt_4, sep='\n')
