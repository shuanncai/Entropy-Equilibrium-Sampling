import json
import random

def sample_jsonl(input_file, output_file, sample_size=200):
    """
    Randomly sample specified number of data from JSONL file
    
    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
        sample_size: Number of samples to extract
    """
    # Read all data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Total original data: {len(data)}")
    
    # Check if data is sufficient
    if len(data) < sample_size:
        print(f"Warning: Original data only has {len(data)} items, less than required {sample_size} items")
        sample_size = len(data)
    
    # Random sampling
    sampled_data = random.sample(data, sample_size)
    
    # Save sampling results
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Successfully sampled {len(sampled_data)} data items, saved to: {output_file}")

# Usage example
if __name__ == "__main__":
    # Set random seed to ensure reproducible results (optional)
    random.seed(42)
    
    # Execute sampling
    sample_jsonl(
        input_file='wikitext_processed/wikitext103_validation.jsonl',
        output_file='wikitext_processed/wikitext103_validation_200.jsonl',
        sample_size=200
    )
