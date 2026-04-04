from typing import List, Dict, Any

from utils import read_json, save_json


def load_and_merge_datasets(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load and merge datasets from multiple file paths.
    
    Args:
        file_paths: List of file paths to load and merge
        
    Returns:
        Merged dataset
    """
    merged_dataset = []
    
    for file_path in file_paths:
        data = read_json(file_path)
        print(f"Loaded {len(data)} samples from {file_path}")
        
        # Process each sample
        for sample in data:
            processed_sample = {
                'pre_text': sample['pre_text'],
                'post_text': sample['post_text'],
                'table': sample['table'],
                'qa': {
                    'question': sample['qa']['question'],
                    'answer': sample['qa']['answer'],
                    'explanation': sample['qa']['explanation'],
                    'steps': sample['qa']['steps'],
                    'program': sample['qa']['program'],
                    'exe_ans': sample['qa']['exe_ans'],
                    'program_re': sample['qa']['program_re']
                },
                'id': sample['id']
            }
            merged_dataset.append(processed_sample)
            
    return merged_dataset


def main():
    """Main function to merge FinQA datasets."""
    # Input file paths
    input_file_paths = [
        'data/finqa/dev.json',
        'data/finqa/test.json',
        'data/finqa/train.json'
    ]

    # Load and merge datasets
    dataset = load_and_merge_datasets(input_file_paths)
    print(f"Total merged samples: {len(dataset)}")

    # Save merged dataset
    output_file_path = 'data/process/0802/finqa.json'
    save_json(output_file_path, dataset)
    print(f"Saved merged dataset to {output_file_path}")


if __name__ == "__main__":
    main()