import re
from typing import List, Dict, Any

from utils import read_json, save_json
from assets.template import prompt


def format_comma_spaces(input_str: str) -> str:
    """
    Format comma spacing in string.
    
    Args:
        input_str: Input string
        
    Returns:
        Formatted string with proper comma spacing
    """
    return re.sub(r',(\s*)', ', ', input_str)


def prepare_user_prompt(
    pre_text: str, 
    table: str, 
    post_text: str, 
    question: str, 
    prompt_template: str
) -> str:
    """
    Prepare user prompt by replacing placeholders.
    
    Args:
        pre_text: Text before table
        table: Table data
        post_text: Text after table
        question: Question to answer
        prompt_template: Template with placeholders
        
    Returns:
        Formatted user prompt
    """
    formatted_prompt = prompt_template.replace('pre_text_placeholder', pre_text)
    formatted_prompt = formatted_prompt.replace('table_placeholder', table)
    formatted_prompt = formatted_prompt.replace('post_text_placeholder', post_text)
    formatted_prompt = formatted_prompt.replace('question_placeholder', question)
    return formatted_prompt


def prepare_assistant_response(program: str, exe_ans: Any) -> str:
    """
    Prepare assistant response.
    
    Args:
        program: Program string
        exe_ans: Execution answer
        
    Returns:
        Formatted assistant response
    """
    return f"{format_comma_spaces(program)}\n{str(exe_ans)}"


def create_message_instance(
    user_content: str, 
    assistant_content: str, 
    sample_id: str
) -> Dict[str, Any]:
    """
    Create a message instance with user and assistant content.
    
    Args:
        user_content: User prompt content
        assistant_content: Assistant response content
        sample_id: Sample identifier
        
    Returns:
        Message instance dictionary
    """
    return {
        'messages': [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant_content}
        ],
        'id': sample_id
    }


def load_datasets(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load datasets from multiple file paths.
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        Combined dataset
    """
    dataset = []
    for file_path in file_paths:
        data = read_json(file_path)
        dataset.extend(data)
        print(f"Loaded {len(data)} samples from {file_path}")
    return dataset


def main():
    """Main function to format benchmark data."""
    # Input file paths
    input_file_paths = [
        'data/process/0802/private_test.json'
    ]

    # Load dataset
    dataset = load_datasets(input_file_paths)
    print(f"Total samples: {len(dataset)}")

    # Process dataset
    formatted_dataset = []
    
    for sample in dataset:
        # Prepare user content
        pre_text = ' '.join(sample['pre_text'])
        table = str(sample['table'])
        post_text = ' '.join(sample['post_text'])
        question = sample['qa']['question']
        
        user_content = prepare_user_prompt(
            pre_text, table, post_text, question, prompt
        )

        # Prepare assistant content (this seems to be a bug in original code)
        # Fixed to actually call the function instead of storing a string
        assistant_content = prepare_assistant_response(
            sample['qa']['program'], sample['qa']['exe_ans']
        )
        
        # Create instance
        instance = create_message_instance(
            user_content, assistant_content, sample['id']
        )
        
        formatted_dataset.append(instance)

    print(f"Formatted samples: {len(formatted_dataset)}")
    
    # Save formatted dataset
    output_file_path = 'data/process/0802/bmark_private_test.json'
    save_json(output_file_path, formatted_dataset)
    print(f"Saved formatted dataset to {output_file_path}")


if __name__ == "__main__":
    main()
