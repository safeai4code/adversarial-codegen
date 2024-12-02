from typing import List, Dict, Any, Optional, Tuple
import re
import random
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

class SynonymAttack:
    """
    Performs synonym replacement on natural language prompts or docstring comments.
    """
    
    def __init__(self, replacement_probability: float = 0.3, max_synonyms: int = 5):
        """
        Initialize the attack.
        
        Args:
            replacement_probability: Chance of replacing each eligible word
            max_synonyms: Maximum number of synonyms to consider for each word
        """
        self.replacement_probability = replacement_probability
        self.max_synonyms = max_synonyms
        self.stop_words = set(stopwords.words('english'))
        # Parts of speech that we want to replace
        self.replaceable_pos = {'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'RB'}
    
    def attack_prompt(self, prompt: str) -> str:
        """Apply synonym replacement to natural language prompt."""
        tokens = word_tokenize(prompt)
        pos_tags = pos_tag(tokens)
        
        modified_tokens = []
        for word, pos in pos_tags:
            if (pos[:2] in self.replaceable_pos and 
                word.lower() not in self.stop_words and
                random.random() < self.replacement_probability):
                synonym = self._find_synonym(word, pos)
                modified_tokens.append(synonym if synonym else word)
            else:
                modified_tokens.append(word)
        
        return self._reconstruct_text(modified_tokens)
    
    def attack_code_comments(self, code: str) -> str:
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
        num_synonyms = min(len(synonyms), self.max_synonyms)
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