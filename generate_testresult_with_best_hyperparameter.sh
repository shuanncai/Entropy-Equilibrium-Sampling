#!/bin/bash

# Optimal hyperparameter multiple runs script - ees decoding method
# Used to run determined optimal hyperparameters multiple times for stable results

# Basic configuration
BASE_CMD="python generate.py"
DECODING_METHOD="ees"
INFILE="data_generate/wikitext103/test_input.jsonl"
MODEL="/path/to/your/model/Llama-3.1-8B-Instruct"
GPUS_PER_MODEL=1
WORLD_SIZE=1
BATCH_SIZE=16
MAX_NEW_TOKENS=256
BEGIN_GPU=0

# Optimal hyperparameter configuration - modify these values based on your best results
TEMPERATURE_T_LIST=(1.0)  # Changed to list format, containing multiple temperature values

# Experiment configuration
NUM_RUNS=5  # Run 5 times for each temperature

# Base output directory
BASE_RESULTS_DIR="./results_llama3.1_8b/wikitext103/ees/test"

# Log file
LOG_FILE="ideal_hyperparameter_runs_$(date +%Y%m%d_%H%M%S).log"

echo "Starting optimal hyperparameter multiple runs experiment..."
echo "Optimal hyperparameter configuration:"
echo "  temperature_t list: ${TEMPERATURE_T_LIST[@]}"
echo "Number of runs per temperature: $NUM_RUNS"
echo "Start time: $(date)"
echo "================================"

# Variables to record successful and failed experiments
success_count=0
total_count=0

# Calculate total number of experiments
total_count=$(( ${#TEMPERATURE_T_LIST[@]} * NUM_RUNS ))

# Iterate through all temperature values
for temperature in "${TEMPERATURE_T_LIST[@]}"; do
    # Create results directory for current temperature
    RESULTS_DIR="${BASE_RESULTS_DIR}/temperature_${temperature}"
    mkdir -p "$RESULTS_DIR"
    
    echo ""
    echo "Currently testing temperature: $temperature"
    echo "Results directory: $RESULTS_DIR"
    echo "------------------------"
    
    # Run NUM_RUNS times for current temperature
    for run in $(seq 1 $NUM_RUNS); do
        echo ""
        echo "Running experiment $run/$NUM_RUNS (temperature=$temperature)..."
        
        # Build output filename
        OUTFILE="${RESULTS_DIR}/test_Llama-3.1-8B-Instruct_ees_temperature_${temperature}_run${run}.jsonl"

        # Build complete command
        FULL_CMD="$BASE_CMD \
            --decoding_method $DECODING_METHOD \
            --infile $INFILE \
            --outfile $OUTFILE \
            --model $MODEL \
            --gpus_per_model $GPUS_PER_MODEL \
            --world_size $WORLD_SIZE \
            --batch_size $BATCH_SIZE \
            --max_new_tokens $MAX_NEW_TOKENS \
            --temperature_t $temperature \
            --begin_gpu $BEGIN_GPU"
        
        echo "Executing command: $FULL_CMD"
        
        # Execute command and record time
        start_time=$(date +%s)
        
        if eval $FULL_CMD; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo "✓ Run $run executed successfully, duration: ${duration}s"
            echo "Output file: $OUTFILE"
            ((success_count++))
        else
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo "✗ Run $run execution failed, duration: ${duration}s"
            echo "Error occurred at: $(date)"
            
            # Ask whether to continue remaining experiments
            remaining=$((total_count - (success_count + (run-1) + (${#TEMPERATURE_T_LIST[@]} - ${TEMPERATURE_T_LIST[@]%%$temperature*} - 1) * NUM_RUNS )))
            if [ $remaining -gt 0 ]; then
                read -p "There are still $remaining experiments remaining, continue? (y/n): " continue_exp
                if [[ $continue_exp != "y" && $continue_exp != "Y" ]]; then
                    echo "User chose to stop experiment"
                    break 2  # Break out of two nested loops
                fi
            fi
        fi
        
        echo "Run $run completed at: $(date)"
        echo "------------------------"
    done
done

echo ""
echo "Optimal hyperparameter multiple runs experiment completed!"
echo "Experiment summary: $success_count/$total_count successful"
echo "End time: $(date)"
echo "Result files saved in: $BASE_RESULTS_DIR"

# Display all generated files
echo ""
echo "Generated result files list:"
find "$BASE_RESULTS_DIR" -name "*.jsonl" -exec ls -la {} \;
