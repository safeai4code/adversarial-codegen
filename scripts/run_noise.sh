#!/bin/bash

# Default values
MODEL_NAME="Llama-3.2-3B"
NOISE_TYPE="gaussian"
MODEL_PATH="/home/wding8/models/${MODEL_NAME}"
OUTPUT_DIR="/home/wding8/RQ1_results/noise_attack"
MAX_PARALLEL=4  # Default maximum parallel processes

# Function to display script usage
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --model_name MODEL_NAME    Model name (default: Llama-3.2-3B)"
  echo "  --noise_type NOISE_TYPE    Noise type: gaussian or uniform (default: gaussian)"
  echo "  --max_parallel N           Maximum number of parallel processes (default: 4)"
  echo "  --help                     Display this help message"
  echo ""
  echo "The script will loop through:"
  echo "  - Quantization: None, 8bit, 4bit"
  echo "  - Noise levels: 1e-4, 1e-3, 3e-3, 5e-3, 1e-2"
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_name)
      MODEL_NAME="$2"
      MODEL_PATH="/home/wding8/models/${MODEL_NAME}"
      shift 2
      ;;
    --noise_type)
      if [[ "$2" == "gaussian" || "$2" == "uniform" ]]; then
        NOISE_TYPE="$2"
        shift 2
      else
        echo "Error: Noise type must be 'gaussian' or 'uniform'"
        usage
      fi
      ;;
    --max_parallel)
      if [[ "$2" =~ ^[0-9]+$ ]]; then
        MAX_PARALLEL="$2"
        shift 2
      else
        echo "Error: Max parallel must be a positive integer"
        usage
      fi
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Create a short abbreviation for the noise type to use in filenames
if [[ "$NOISE_TYPE" == "gaussian" ]]; then
  NOISE_ABBR="gs"
else
  NOISE_ABBR="un"
fi

# Array of quantization options to loop through
QUANTIZATIONS=("None" "8bit" "4bit")

# Array of noise levels to loop through
NOISE_LEVELS=("1e-4" "1e-3" "3e-3" "5e-3" "1e-2")

# Create a log directory
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Create a temp directory for job tracking
TEMP_DIR="${OUTPUT_DIR}/tmp"
mkdir -p "$TEMP_DIR"

# Create a file to track running jobs
TRACKING_FILE="${TEMP_DIR}/running_jobs.txt"
> "$TRACKING_FILE"  # Empty the file

# Function to count currently running jobs
count_running_jobs() {
  # Count lines in the tracking file that contain PIDs of running processes
  local count=0
  local to_remove=()
  
  # If file doesn't exist yet, return 0
  if [[ ! -f "$TRACKING_FILE" ]]; then
    return 0
  fi
  
  # Read the file line by line
  while IFS= read -r line; do
    local pid=$(echo "$line" | awk '{print $1}')
    
    # Check if the process is still running
    if ps -p "$pid" >/dev/null 2>&1; then
      ((count++))
    else
      # Mark this line for removal
      to_remove+=("$line")
    fi
  done < "$TRACKING_FILE"
  
  # Remove completed jobs from tracking file
  if [[ ${#to_remove[@]} -gt 0 ]]; then
    for job in "${to_remove[@]}"; do
      # Use sed to remove the exact line
      sed -i "\|^$job$|d" "$TRACKING_FILE"
      echo "Job completed: $job"
    done
  fi
  
  return $count
}

# Function to wait for an available slot
wait_for_slot() {
  while true; do
    count_running_jobs
    local running=$?
    
    if [[ $running -lt $MAX_PARALLEL ]]; then
      return
    fi
    
    echo "Maximum parallel jobs ($MAX_PARALLEL) reached. Currently running: $running. Waiting..."
    sleep 10
  done
}

# Function to run evaluation
run_evaluation() {
  local quant="$1"
  local noise_level="$2"
  
  # Prepare output file path with appropriate naming convention
  if [[ "$quant" == "None" ]]; then
    quant_suffix="base"
    quant_arg=""
  else
    quant_suffix="${quant:0:1}bnb"
    quant_arg="--quantization $quant"
  fi
  
  local job_id="${MODEL_NAME}-${quant_suffix}-${NOISE_ABBR}-${noise_level}"
  output_path="${OUTPUT_DIR}/${job_id}/results.jsonl"
  log_file="${LOG_DIR}/${job_id}.log"
  
  # Create output directory if it doesn't exist
  mkdir -p "$(dirname "$output_path")"
  
  # Log the command that will be executed
  echo "Starting job: $job_id"
  
  # Create job script for this specific task
  local job_script="${LOG_DIR}/job_${job_id}.sh"
  cat > "$job_script" << EOF
#!/bin/bash
echo "Starting job at \$(date)" > "$log_file"
echo "Command: adversarial-codegen-noise evaluate_single_model --model_path $MODEL_PATH $quant_arg --output_path $output_path --noise_type $NOISE_TYPE --noise_level $noise_level" >> "$log_file"

adversarial-codegen-noise evaluate_single_model \\
  --model_path $MODEL_PATH \\
  $quant_arg \\
  --output_path $output_path \\
  --noise_type $NOISE_TYPE \\
  --noise_level $noise_level \\
  >> "$log_file" 2>&1

echo "Job completed at \$(date)" >> "$log_file"

# Remove this job from the tracking file when done
sed -i "\|^[0-9]\\+ $job_id$|d" "$TRACKING_FILE"
EOF

  chmod +x "$job_script"
  
  # Run the job script in background using nohup
  nohup "$job_script" > /dev/null 2>&1 &
  local pid=$!
  
  # Verify that the process is actually running
  sleep 1
  if ! ps -p "$pid" >/dev/null 2>&1; then
    echo "WARNING: Process for $job_id failed to start or terminated immediately!"
    return 1
  fi
  
  # Add to the tracking file
  echo "$pid $job_id" >> "$TRACKING_FILE"
  
  echo "  PID: $pid"
  echo "  Output: $output_path"
  echo "  Log: $log_file"
  
  return 0
}

# Main execution
echo "Starting evaluation for model: $MODEL_NAME with noise type: $NOISE_TYPE"
echo "Will loop through quantizations: ${QUANTIZATIONS[*]}"
echo "Will loop through noise levels: ${NOISE_LEVELS[*]}"
echo "Maximum parallel processes: $MAX_PARALLEL"
echo "----------------------------------------"

# Loop through each quantization and noise level combination
total_jobs=$((${#QUANTIZATIONS[@]} * ${#NOISE_LEVELS[@]}))
completed_jobs=0

for quant in "${QUANTIZATIONS[@]}"; do
  for noise_level in "${NOISE_LEVELS[@]}"; do
    # Wait until we have fewer than MAX_PARALLEL running jobs
    wait_for_slot
    
    # Run the evaluation
    run_evaluation "$quant" "$noise_level"
    
    # Count jobs
    ((completed_jobs++))
    echo "Scheduled job $completed_jobs/$total_jobs"
    echo "----------------------------------------"
    
    # Brief pause to ensure process starts properly
    sleep 1
  done
done

echo "All jobs scheduled. Waiting for remaining jobs to complete..."

# Wait until all jobs are done
while true; do
  count_running_jobs
  running=$?
  
  if [[ $running -eq 0 ]]; then
    break
  fi
  
  echo "Still waiting on $running process(es)..."
  sleep 15
done

echo "All evaluations completed for $MODEL_NAME with noise type $NOISE_TYPE"