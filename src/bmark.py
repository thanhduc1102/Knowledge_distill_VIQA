import argparse
import re
import concurrent.futures
from typing import Dict, Any, Tuple, List, Optional
from openai import OpenAI
from tqdm import tqdm

from utils import read_json, save_json
from program_tokenizer import program_tokenization


def obtain_one_sample(
    sample: Dict[str, Any], 
    base_url: str, 
    model: str, 
    max_retries: int = 4
) -> Dict[str, Any]:
    """
    Process a single sample with the LLM API.
    
    Args:
        sample: Input data sample
        base_url: API base URL
        model: Model name
        max_retries: Maximum number of retry attempts
        
    Returns:
        Processed sample with content and reasoning
    """
    client = OpenAI(base_url=base_url, api_key="no-need")
    
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[sample['messages'][0]],
                temperature=0.6,
                top_p=0.95
            )

            content = completion.choices[0].message.content
            reasoning_content = getattr(completion.choices[0].message, 'reasoning_content', '')
            
            if content or reasoning_content:
                sample['content'] = content
                sample['reasoning_content'] = reasoning_content
                return sample
            else:
                print('Error: Empty response from model')
                raise ValueError("Empty completion.choices[0].message")

        except Exception as e:
            print(f'Retry attempt {attempt+1} failed: {str(e)}')
            if attempt == max_retries:
                print("Max retries exceeded")
                return sample


def process_dataset(
    dataset: List[Dict[str, Any]], 
    base_url: str, 
    model: str, 
    max_workers: int
) -> List[Dict[str, Any]]:
    """
    Process entire dataset using thread pool executor.
    
    Args:
        dataset: List of data samples
        base_url: API base URL
        model: Model name
        max_workers: Number of concurrent workers
        
    Returns:
        List of processed samples
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(obtain_one_sample, sample, base_url, model) 
            for sample in dataset
        ]
        
        # Process completed tasks with progress bar
        with tqdm(total=len(dataset), desc="Processing samples") as progress_bar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Processing error: {str(e)}")
                finally:
                    progress_bar.update(1)
                    
    return results


def extract_program_and_answer(ground_truth_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract program and answer from ground truth string.
    
    Args:
        ground_truth_str: Ground truth string containing program and answer
        
    Returns:
        Tuple of (answer, program)
    """
    # Extract program
    program_matches = list(re.finditer(
        r"\*\*Chương trình tính toán:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", 
        ground_truth_str
    ))
    program = program_matches[-1].group(1).strip() if program_matches else None
    
    # Extract answer
    answer_matches = list(re.finditer(
        r"\*\*Đáp án cuối cùng:\*\*\s*((?:.|\n)*?)(?=\s*\*\*|$)", 
        ground_truth_str
    ))
    answer = answer_matches[-1].group(1).strip() if answer_matches else None
    
    return answer, program


def check_answer_match(
    ground_truth: str, 
    prediction: str
) -> Tuple[int, int]:
    """
    Check if ground truth and prediction match.
    
    Args:
        ground_truth: Ground truth string
        prediction: Prediction string
        
    Returns:
        Tuple of (execution_match, program_match)
    """
    exe_gt, pro_gt = extract_program_and_answer(ground_truth)
    exe_pr, pro_pr = extract_program_and_answer(prediction)
    
    execution_match = 1 if exe_gt == exe_pr else 0
    program_match = 1 if pro_gt == pro_pr else 0
    
    return execution_match, program_match


def evaluate_results(
    output_file_path: str, 
    data: List[Dict[str, Any]]
) -> None:
    """
    Evaluate model results and save to file.
    
    Args:
        output_file_path: Path to save results
        data: List of processed samples
    """
    results = []
    execution_matches = 0
    program_matches = 0
    
    for sample in data:
        ground_truth = sample['messages'][1]['content']
        
        # Skip invalid samples
        if ('content' not in sample) or not isinstance(sample['content'], str):
            print("Warning: Invalid sample content")
            continue
            
        prediction = sample['content']
        exe_pr, pro_pr = extract_program_and_answer(prediction)
        
        try:
            results.append({
                'id': sample['id'],
                'predicted': program_tokenization(pro_pr)
            })
        except Exception as e:
            print(f"Tokenization error: {e}")
        
        execution_match, program_match = check_answer_match(ground_truth, prediction)
        execution_matches += execution_match
        program_matches += program_match
    
    # Calculate metrics
    execution_accuracy = execution_matches / len(data) if data else 0
    program_accuracy = program_matches / len(data) if data else 0
    
    print(f"Execution Accuracy: {execution_accuracy:.2%}")
    print(f"Program Accuracy: {program_accuracy:.2%}")
    
    save_json(output_file_path, results)


def main():
    """Main function to process and evaluate dataset."""
    parser = argparse.ArgumentParser(description='Process dataset with OpenAI API')
    parser.add_argument('--ifp', required=True, help='Input file path')
    parser.add_argument('--ofp_predictions', required=True, help='Output predictions file path')
    parser.add_argument('--ofp_results', required=True, help='Output results file path')
    parser.add_argument('--base_url', default="http://localhost:8000/v1", help='API base URL')
    parser.add_argument('--model', default="qwen3", help='Model name')
    parser.add_argument('--max_workers', type=int, default=64, help='Max concurrent workers')
    
    args = parser.parse_args()

    # Load dataset
    dataset = read_json(args.ifp)
    print(f"Loaded {len(dataset)} samples")
    
    # Process dataset
    processed_data = process_dataset(
        dataset=dataset,
        base_url=args.base_url,
        model=args.model,
        max_workers=args.max_workers
    )
    
    # Save predictions
    save_json(args.ofp_predictions, processed_data)
    print(f"Saved {len(processed_data)} items to {args.ofp_predictions}")

    # Evaluate results
    evaluate_results(args.ofp_results, processed_data)


if __name__ == "__main__":
    main()