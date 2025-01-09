import re

from typing import Optional, List, Tuple


def extract_single_function(content: str) -> Optional[str]:
    """
    Extract a single Python function definition from the given content, including decorators,
    docstrings, and the complete function body.
    
    Args:
        content (str): The text content containing a Python function
        
    Returns:
        str: The extracted function definition, or None if no function found
        
    Example:
        >>> code = '''
        ... @decorator
        ... def example(x: int) -> str:
        ...     ""Docstring.""
        ...     return str(x)
        ... '''
        >>> print(extract_single_function(code))
        @decorator
        def example(x: int) -> str:
            ""Docstring.""
            return str(x)
    """
    # First, clean the content by removing code fence markers if present
    content = remove_code_fences(content)
    
    # Find all potential function blocks (including decorators)
    function_blocks = find_function_blocks(content)
    
    if not function_blocks:
        return None
        
    # Return the first complete function block found
    return function_blocks[0] if function_blocks else None


def remove_code_fences(content: str) -> str:
    """Remove markdown code fence markers from the content."""
    lines = content.split('\n')
    # Remove ```python, ```, and similar markers
    cleaned_lines = [line for line in lines if not line.strip().startswith('```')]
    return '\n'.join(cleaned_lines)


def find_function_blocks(content: str) -> List[str]:
    """
    Find all function blocks in the content, including decorators and function body.
    """
    # Pattern to match decorator lines
    decorator_pattern = r'@[\w\.]+'
    
    # Pattern to match function definition
    func_def_pattern = r'def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?:'
    
    # Split content into lines for processing
    lines = content.split('\n')
    functions = []
    current_block = []
    in_function = False
    base_indent = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check for decorator
        if re.match(decorator_pattern, stripped):
            if not in_function:
                current_block = [line]
                i += 1
                continue
                
        # Check for function definition
        if re.match(func_def_pattern, stripped):
            in_function = True
            if not current_block:
                current_block = []
            current_block.append(line)
            base_indent = len(line) - len(stripped)
            i += 1
            
            # Collect the function body
            while i < len(lines):
                next_line = lines[i]
                if not next_line.strip():  # Handle empty lines
                    current_block.append(next_line)
                    i += 1
                    continue
                    
                # Check if we're still in the function body by comparing indentation
                curr_indent = len(next_line) - len(next_line.lstrip())
                if curr_indent <= base_indent and next_line.strip():
                    # We've reached the end of the function
                    break
                    
                current_block.append(next_line)
                i += 1
            
            # Clean up trailing empty lines
            while current_block and not current_block[-1].strip():
                current_block.pop()
                
            if current_block:
                functions.append('\n'.join(current_block))
            current_block = []
            in_function = False
            continue
            
        i += 1
    
    return functions


def extract_functions(content: str) -> List[str]:
    """
    Extract all Python function definitions from the given content.
    
    Args:
        content (str): The text content containing Python functions
        
    Returns:
        List[str]: List of extracted function definitions
    """
    return '\n'.join(find_function_blocks(content))
