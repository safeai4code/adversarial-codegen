import re
from typing import List, Optional, Tuple


def extract_functions(content: str) -> str:
    """
    Extract code blocks while handling indentation and filtering unnecessary imports.
    
    Args:
        content (str): The text content containing Python code
        
    Returns:
        str: The cleaned and properly formatted code block
    """
    # First, clean the content by removing code fence markers if present
    content = remove_code_fences(content)
    
    # Normalize line endings
    content = normalize_content(content)
    
    # Split content into lines and get base indentation
    lines = content.split('\n')
    base_indent = get_base_indent(lines)
    
    # Remove the base indentation from all lines
    lines = [remove_indent(line, base_indent) for line in lines]
    content = '\n'.join(lines)
    
    # Extract only necessary elements
    functions = find_function_blocks(content)
    necessary_imports = find_necessary_imports(content)
    
    # Combine the elements
    result = []
    if necessary_imports:
        result.extend(necessary_imports)
        if functions:
            result.append("")  # Add blank line between imports and functions
    
    if functions:
        result.extend(functions)
    
    return '\n'.join(result)


def get_base_indent(lines: List[str]) -> int:
    """
    Get the base indentation level from the code.
    """
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return 0
    
    # Find minimum indentation of non-empty lines
    indents = [len(line) - len(line.lstrip()) for line in non_empty_lines]
    return min(indents) if indents else 0


def remove_indent(line: str, base_indent: int) -> str:
    """
    Remove base indentation from a line while preserving relative indentation.
    """
    if not line.strip():
        return line
    
    current_indent = len(line) - len(line.lstrip())
    if current_indent >= base_indent:
        return line[base_indent:]
    return line


def find_necessary_imports(content: str) -> List[str]:
    """
    Find necessary imports (those not inside if __name__ == '__main__' block).
    """
    # import_pattern = r'^(?:from\s+[\w.]+\s+import\s+(?:[\w.]+(?:\s+as\s+\w+)?(?:\s*,
    # \s*[\w.]+(?:\s+as\s+\w+)?)*|[*])|import\s+(?:[\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*))\s*$'
    import_pattern = (
        r'^(?:from\s+[\w.]+\s+import\s+(?:[\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+'
        r'(?:\s+as\s+\w+)?)*|[*])|import\s+(?:[\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*))\s*$'
    )
    main_block_pattern = r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:'
    
    lines = content.split('\n')
    imports = []
    in_main_block = False
    main_block_indent = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check for main block
        if re.match(main_block_pattern, stripped):
            in_main_block = True
            main_block_indent = len(line) - len(line.lstrip())
            continue
        
        # Check if we're still in main block
        if in_main_block:
            current_indent = len(line) - len(line.lstrip())
            if not stripped or current_indent > main_block_indent:
                continue
            else:
                in_main_block = False
        
        # Collect imports outside main block
        if not in_main_block and re.match(import_pattern, stripped):
            imports.append(line.rstrip())
    
    return imports


def remove_code_fences(content: str) -> str:
    """Remove markdown code fence markers from the content."""
    lines = content.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith('```')]
    return '\n'.join(cleaned_lines)


def normalize_content(content: str) -> str:
    """Normalize content by fixing line endings and spacing."""
    content = content.replace('\r\n', '\n')
    content = re.sub(r'"""(.*?)"""', lambda m: f'"""{m.group(1)}"""\n', content, flags=re.DOTALL)
    content = re.sub(r'([^\n])(\s*def\s+)', r'\1\n\2', content)
    return content


def find_function_blocks(content: str) -> List[str]:
    """Find all function blocks including decorators and docstrings."""
    decorator_pattern = r'@[\w\.]+'
    func_def_pattern = r'def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?:'
    
    lines = content.split('\n')
    functions = []
    current_block = []
    in_function = False
    base_indent = None
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        
        # Skip empty lines before function
        if not stripped and not in_function:
            i += 1
            continue
        
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
            
            # Look backward for docstring
            if not any('"""' in block for block in current_block):
                doc_start = i - 1
                while doc_start >= 0 and '"""' in lines[doc_start]:
                    current_block.insert(0, lines[doc_start].rstrip())
                    doc_start -= 1
            
            current_block.append(line)
            base_indent = len(line) - len(stripped)
            i += 1
            
            # Collect function body
            while i < len(lines):
                next_line = lines[i].rstrip()
                if not next_line.strip():
                    current_block.append(next_line)
                    i += 1
                    continue
                
                curr_indent = len(next_line) - len(next_line.lstrip())
                if curr_indent <= base_indent and next_line.strip():
                    break
                
                current_block.append(next_line)
                i += 1
            
            # Clean trailing empty lines
            while current_block and not current_block[-1].strip():
                current_block.pop()
            
            if current_block:
                functions.append('\n'.join(current_block))
            current_block = []
            in_function = False
            continue
        
        i += 1
    
    return functions


if __name__ == "__main__":
    # Test case with imports
    code = '''
    import re
    from typing import List, Optional

    """
    Write a function to count the number of occurence of the string'std' in a given string.
    assert count_occurance("letstdlenstdporstd") == 3
    """
    def count_occurance(string):
        count = 0
        for i in range(len(string)):
            if string[i:i+3] == "std":
                count += 1
        return count

    def count_occurance(string):
        count = 0
        for i in range(len(string)):
            if string[i:i+3] == "std":
                count += 1
        return count
    '''
    
    result = extract_functions(code)
    print(result)
