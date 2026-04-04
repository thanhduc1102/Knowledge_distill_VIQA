def program_tokenization(original_program: str) -> list:
    """
    Tokenize a program string into structured tokens.
    
    Args:
        original_program: Program string to tokenize
        
    Returns:
        List of tokens
    """
    program = []
    current_token = ''
    bracket_level = 0
    index = 0
    
    while index < len(original_program):
        char = original_program[index]
        
        if char == '(':
            bracket_level += 1
            if bracket_level == 1:
                current_token += char
                program.append(current_token.strip())
                current_token = ''
            else:
                current_token += char
        elif char == ')':
            if bracket_level == 1:
                if current_token.strip() != '':
                    program.append(current_token.strip())
                    current_token = ''
                program.append(')')
            else:
                current_token += char
            bracket_level -= 1
        elif char == ',':
            if bracket_level == 0:
                if current_token.strip() != '':
                    program.append(current_token.strip())
                    current_token = ''
            elif bracket_level == 1:
                if current_token.strip() != '':
                    program.append(current_token.strip())
                    current_token = ''
            else:
                current_token += char
        else:
            current_token += char
            
        index += 1
        
    # Add remaining token if it exists
    if current_token.strip() != '':
        program.append(current_token.strip())
        
    # Add end-of-file marker
    program.append('EOF')
    
    return program


# Example usage:
# program1 = "add(6851, 9091), add(#0, 14606)"
# tokens = program_tokenization(program1)
# print(tokens)