#!/bin/bash

# Batch evaluation script - supports multiple temperature values
# Configuration parameters
MODEL_PATH="/path/to/your/model/Llama-3.1-8B-Instruct"
MODEL_NAME="Llama-3.1-8B-Instruct"
TASK_NAME="strategy_qa"

# Temperature parameter list
TEMPERATURE_VALUES=(1.0)

# Number of runs
RUNS=(1 2 3 4 5)

echo "Starting batch evaluation..."
echo "Model: $MODEL_PATH"
echo "Task: $TASK_NAME"
echo "Temperature values: ${TEMPERATURE_VALUES[@]}"
echo "Number of runs: ${RUNS[@]}"
echo ""

# Statistics variables
total_jobs=0
completed_jobs=0
failed_jobs=0

# Calculate total number of tasks
total_jobs=$((${#TEMPERATURE_VALUES[@]} * ${#RUNS[@]}))

echo "Total tasks to run: $total_jobs"
echo "================================"

# Double nested loop to execute all combinations
for temperature in "${TEMPERATURE_VALUES[@]}"; do
    echo "📊 Processing temperature: $temperature"
    BASE_DIR="results_llama3.1_8b/strategy_qa/my_entropy/test/temperature_${temperature}_0.20"
    
    for run in "${RUNS[@]}"; do
        # Build file paths
        input_file="${BASE_DIR}/test_${MODEL_NAME}_my_entropy_temperature_${temperature}_run${run}.jsonl"
        output_file="${BASE_DIR}/results.jsonl"
        
        echo "Processing: temperature=${temperature}, run=${run}"
        echo "Input file: $input_file"
        echo "Output file: $output_file"
        
        # Check if input file exists
        if [ ! -f "$input_file" ]; then
            echo "❌ Warning: Input file does not exist, skipping"
            ((failed_jobs++))
            echo ""
            continue
        fi
        
        # Create output directory if it doesn't exist
        mkdir -p "$(dirname "$output_file")"
        
        # Execute evaluation command
        echo "🚀 Starting execution..."
        if python evaluation.py \
            --model "$MODEL_PATH" \
            --task_name "$TASK_NAME" \
            --load_generations_path "$input_file" \
            --metric_output_path "$output_file" \
            --diversity \
            --only_calculate_correct_case; then
            echo "✅ Completed: temperature=${temperature}, run=${run}"
            ((completed_jobs++))
        else
            echo "❌ Failed: temperature=${temperature}, run=${run}"
            ((failed_jobs++))
        fi
        
        echo "Progress: $((completed_jobs + failed_jobs))/$total_jobs"
        echo "--------------------------------"
    done
    echo "🎯 Temperature $temperature processing completed"
    echo "================================"
done

# Output final statistics
echo ""
echo "Batch evaluation completed!"
echo "Total tasks: $total_jobs"
echo "Successfully completed: $completed_jobs"
echo "Failed/skipped: $failed_jobs"
echo "Success rate: $(( completed_jobs * 100 / total_jobs ))%"
echo ""
