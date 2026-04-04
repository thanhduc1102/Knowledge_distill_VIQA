import re
from typing import List, Dict, Any

from utils import read_json, save_json
from assets.think import prompt


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
    prompt_template: str, 
    program: str, 
    exe_ans: Any
) -> str:
    """
    Prepare user prompt by replacing placeholders.
    
    Args:
        pre_text: Text before table
        table: Table data
        post_text: Text after table
        question: Question to answer
        prompt_template: Template with placeholders
        program: Program string
        exe_ans: Execution answer
        
    Returns:
        Formatted user prompt
    """
    formatted_prompt = prompt_template.replace('pre_text_placeholder', pre_text)
    formatted_prompt = formatted_prompt.replace('table_placeholder', table)
    formatted_prompt = formatted_prompt.replace('post_text_placeholder', post_text)
    formatted_prompt = formatted_prompt.replace('question_placeholder', question)
    formatted_prompt = formatted_prompt.replace('cttt_placeholder', format_comma_spaces(program))
    formatted_prompt = formatted_prompt.replace('dacc_placeholder', str(exe_ans))
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
    return f"**Chương trình tính toán:**\n{format_comma_spaces(program)}\n\n**Đáp án cuối cùng:**\n{str(exe_ans)}"


def create_training_instance(
    user_content: str, 
    assistant_content: str, 
    sample_id: str
) -> Dict[str, Any]:
    """
    Create a training instance.
    
    Args:
        user_content: User prompt content
        assistant_content: Assistant response content
        sample_id: Sample identifier
        
    Returns:
        Training instance dictionary
    """
    return {
        'data_source': 'openai/gsm8k',
        'prompt': [{'content': user_content, 'role': 'user'}],
        'reward_model': {'ground_truth': assistant_content},
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


def process_samples(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all samples to create training instances.
    
    Args:
        dataset: Input dataset
        
    Returns:
        List of processed training instances
    """
    processed_instances = []
    revised_program_count = 0
    
    for sample in dataset:
        # Prepare user content
        pre_text = ' '.join(sample['pre_text'])
        table = str(sample['table'])
        post_text = ' '.join(sample['post_text'])
        question = sample['qa']['question']
        
        user_content = prepare_user_prompt(
            pre_text, table, post_text, question, prompt, 
            sample['qa']['program'], sample['qa']['exe_ans']
        )

        # Prepare assistant content (ground truth)
        assistant_content = prepare_assistant_response(
            sample['qa']['program'], sample['qa']['exe_ans']
        )
        
        # Create instance
        instance = create_training_instance(
            user_content, assistant_content, sample['id']
        )
        processed_instances.append(instance)
        
        # Add additional instance if program_re exists and differs from program
        if 'program_re' in sample['qa']:
            if format_comma_spaces(sample['qa']['program_re']) != format_comma_spaces(sample['qa']['program']):
                revised_program_count += 1
                
                # Prepare user content with revised program
                user_content_re = prepare_user_prompt(
                    pre_text, table, post_text, question, prompt,
                    sample['qa']['program_re'], sample['qa']['exe_ans']
                )
                
                # Prepare assistant content with revised program
                assistant_content_re = prepare_assistant_response(
                    sample['qa']['program_re'], sample['qa']['exe_ans']
                )
                
                # Create instance with revised program
                instance_re = create_training_instance(
                    user_content_re, assistant_content_re, sample['id']
                )
                processed_instances.append(instance_re)
                
    print(f"Samples with revised programs: {revised_program_count}")
    return processed_instances


def main():
    """Main function to format thinking process training data."""
    # Input file paths
    input_file_paths = [
        'data/receive/valid.json',
        'data/receive/train.json',
        # 'data/receive/test.json',
        # 'data/process/0802/finqa.json',
    ]

    # Load dataset
    dataset = load_datasets(input_file_paths)
    print(f"Total samples: {len(dataset)}")

    # Process samples
    processed_dataset = process_samples(dataset)
    print(f"Processed instances: {len(processed_dataset)}")

    # Save dataset
    output_file_path = 'data/process/0802/valid.json'
    save_json(output_file_path, processed_dataset)
    print(f"Saved formatted dataset to {output_file_path}")


if __name__ == "__main__":
    main()
