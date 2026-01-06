import json
import random

def create_small_dataset(sample_ratio=0.2, seed=42):
    """Create small dataset"""
    random.seed(seed)
    
    # Read data
    with open('data_generate/commonsense_qa/train_input.jsonl', 'r', encoding='utf-8') as f:
        input_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    with open('data_generate/commonsense_qa/train.jsonl', 'r', encoding='utf-8') as f:
        train_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    # Random sampling
    total_samples = len(input_data)
    sample_size = int(total_samples * sample_ratio)
    sample_indices = random.sample(range(total_samples), sample_size)
    
    # Extract data
    sampled_input = [input_data[i] for i in sample_indices]
    sampled_train = [train_data[i] for i in sample_indices]
    
    # Save files
    with open('data_generate/commonsense_qa/validation_input.jsonl', 'w', encoding='utf-8') as f:
        for item in sampled_input:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open('data_generate/commonsense_qa/validation.jsonl', 'w', encoding='utf-8') as f:
        for item in sampled_train:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Sampled {sample_size} data points from {total_samples} total data points")

# Run
create_small_dataset()
