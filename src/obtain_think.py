import re
import concurrent.futures
from typing import Dict, Any, Tuple, List, Optional
from openai import OpenAI
from tqdm import tqdm

from utils import read_json, save_json


class ThinkingDataGenerator:
    """Generator for thinking process data using LLMs."""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "qwen3"):
        """
        Initialize the generator.
        
        Args:
            base_url: API base URL
            model: Model name
        """
        self.client = OpenAI(base_url=base_url, api_key="no")
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def extract_program_and_answer(self, ground_truth_str: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract program and answer from ground truth string.
        
        Args:
            ground_truth_str: Ground truth string containing program and answer
            
        Returns:
            Tuple of (program, answer)
        """
        # Extract program
        program_matches = list(re.finditer(
            r"\\*\\*Chương trình tính toán:\\*\\*\\s*((?:.|\\n)*?)(?=\\s*\\*\\*|$)", 
            ground_truth_str
        ))
        program = program_matches[-1].group(1).strip() if program_matches else None
        
        # Extract answer
        answer_matches = list(re.finditer(
            r"\\*\\*Đáp án cuối cùng:\\*\\*\\s*((?:.|\\n)*?)(?=\\s*\\*\\*|$)", 
            ground_truth_str
        ))
        answer = answer_matches[-1].group(1).strip() if answer_matches else None
        
        return program, answer

    def check_answer_match(self, ground_truth: str, prediction: str) -> bool:
        """
        Check if ground truth and prediction match.
        
        Args:
            ground_truth: Ground truth string
            prediction: Prediction string
            
        Returns:
            True if they match, False otherwise
        """
        program_gt, answer_gt = self.extract_program_and_answer(ground_truth)
        program_pr, answer_pr = self.extract_program_and_answer(prediction)
        
        return program_gt == program_pr and answer_gt == answer_pr

    def process_sample(self, sample: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
        """
        Process a single sample with the LLM.
        
        Args:
            sample: Input sample
            max_retries: Maximum number of retry attempts
            
        Returns:
            Processed sample
        """
        for attempt in range(max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": sample['prompt'][0]['content']}],
                    temperature=0.6,
                    top_p=0.95
                )

                # Update token counts
                self.prompt_tokens += completion.usage.prompt_tokens
                self.completion_tokens += completion.usage.completion_tokens
                self.total_tokens += completion.usage.total_tokens
                
                content = completion.choices[0].message.content
                reasoning_content = completion.choices[0].message.reasoning_content
                
                # Check if answer matches ground truth
                if self.check_answer_match(sample['reward_model']['ground_truth'], content):
                    sample['content'] = content
                    sample['reasoning_content'] = reasoning_content
                    sample['token'] = (
                        completion.usage.prompt_tokens, 
                        completion.usage.completion_tokens, 
                        completion.usage.total_tokens
                    )
                    return sample
                else:
                    print('-' * 100)
                    # Extract and print prompt
                    prompt_start = "Hãy **SỬ DỤNG CHÍNH XÁC** nội dung dưới đây trong câu trả lời của bạn:"
                    prompt_index = sample['prompt'][0]['content'].find(prompt_start)
                    if prompt_index != -1:
                        print({"prompt": f"{sample['prompt'][0]['content'][prompt_index + len(prompt_start):]}"})
                    else:
                        print({"prompt": sample['prompt'][0]['content']})
                        
                    print('+' * 25)
                    print({"ground_truth": sample['reward_model']['ground_truth']})
                    print('+' * 25)
                    
                    # Extract and print prediction
                    program_start = "**Chương trình tính toán:**"
                    program_index = content.rfind(program_start)
                    if program_index != -1:
                        print({"predict": content[program_index:]})
                    else:
                        print({"predict": content})
                        
                    raise ValueError("Answer mismatch")

            except Exception as e:
                if attempt == max_retries:
                    print(f"Max retries exceeded")
                    sample['content'] = content
                    sample['reasoning_content'] = reasoning_content
                    sample['token'] = (
                        completion.usage.prompt_tokens, 
                        completion.usage.completion_tokens, 
                        completion.usage.total_tokens
                    )
                    sample['error'] = 1
                    return sample

    def process_dataset(self, dataset: List[Dict[str, Any]], max_workers: int = 64) -> List[Dict[str, Any]]:
        """
        Process entire dataset using thread pool executor.
        
        Args:
            dataset: List of data samples
            max_workers: Number of concurrent workers
            
        Returns:
            List of processed samples
        """
        print('--- REWRITE ---')
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(self.process_sample, sample) for sample in dataset]
            
            # Process completed tasks with progress bar
            with tqdm(total=len(dataset), desc="Processing samples") as progress_bar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Error processing sample: {e}")
                    finally:
                        progress_bar.update(1)
                        
        return results

    def print_token_stats(self) -> None:
        """Print token usage statistics."""
        print(f"Prompt tokens: {self.prompt_tokens}")
        print(f"Completion tokens: {self.completion_tokens}")
        print(f"Total tokens: {self.total_tokens}")


def main():
    """Main function to generate thinking process data."""
    # Initialize generator
    generator = ThinkingDataGenerator()
    
    # Input file path
    input_file_path = 'data/process/0802/valid.json'
    print(f"Processing file: {input_file_path}")
    
    # Load dataset
    data = read_json(input_file_path)
    print(f"Loaded {len(data)} samples")
    
    # Process dataset
    # data = data[:100]  # for debugging
    processed_dataset = generator.process_dataset(data)
    print(f"Processed {len(processed_dataset)} samples")
    
    # Save results
    output_file_path = 'data/process/0802/valid_qwen3.json'
    save_json(output_file_path, processed_dataset)
    print(f"Saved results to {output_file_path}")
    
    # Print token statistics
    generator.print_token_stats()
    print("Processing complete")


if __name__ == "__main__":
    main()