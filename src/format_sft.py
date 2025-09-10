import re
from typing import List, Dict, Any

from utils import read_json, save_json
from assets.template import prompt


def clean_formatting_markers(text: str) -> str:
    """
    Clean formatting markers in text.
    
    Args:
        text: Input text with formatting markers
        
    Returns:
        Cleaned text
    """
    keyword = "**Chương trình tính toán:**"
    last_index = text.rfind(keyword)
    
    # Return original if keyword not found
    if last_index == -1:
        return text
    
    # Keep prefix unchanged
    prefix = text[:last_index]
    # Process suffix
    suffix = text[last_index:]
    
    # Clean ** markers: remove extra spaces around
    suffix = re.sub(r'\*\* +', '**', suffix)  # Remove spaces after **
    suffix = re.sub(r' +\*\*', '**', suffix)  # Remove spaces before **
    
    # Clean newlines: remove extra whitespace
    suffix = re.sub(r'[ \t]+\n', '\n', suffix)  # Whitespace + \n → \n
    suffix = re.sub(r'\n[ \t]+', '\n', suffix)   # \n + whitespace → \n
    
    return prefix + suffix


def remove_instruction_suffix(text: str) -> str:
    """
    Remove instruction suffix from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with suffix removed
    """
    keyword = "\n\nHãy **SỬ DỤNG CHÍNH XÁC** nội dung dưới đây trong câu trả lời của bạn"
    index = text.find(keyword)
    
    if index != -1:
        return text[:index]
    
    print('Warning: Instruction suffix not found')
    return text


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
    Process all samples to create SFT training instances.
    
    Args:
        dataset: Input dataset
        
    Returns:
        List of processed training instances
    """
    processed_instances = []
    
    for sample in dataset:
        # Skip samples with errors
        if 'error' in sample:
            continue
            
        # Process user content
        user_content = remove_instruction_suffix(sample['prompt'][0]['content'])
        
        # Process assistant content
        assistant_content = clean_formatting_markers(sample['content'])
        
        # Create instance
        instance = {
            'messages': [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': assistant_content}
            ],
            'id': sample['id']
        }
        
        processed_instances.append(instance)
        
    return processed_instances


def main():
    """Main function to format SFT training data."""
    # Input file paths
    input_file_paths = [
        'data/process/0802/train_qwen.json',
        # 'data/process/0802/train_en_qwen3.json',
        # 'data/process/0802/valid_qwen3.json'
    ]

    # Load dataset
    dataset = load_datasets(input_file_paths)
    print(f"Total samples: {len(dataset)}")

    # Process samples
    processed_dataset = process_samples(dataset)
    print(f"Processed samples: {len(processed_dataset)}")

    # Save dataset
    output_file_path = 'data/process/0802/train_qwen_sft.json'
    save_json(output_file_path, processed_dataset)
    print(f"Saved formatted dataset to {output_file_path}")


if __name__ == "__main__":
    main()