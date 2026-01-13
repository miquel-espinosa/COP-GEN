#!/bin/bash

# === Configuration ===
JOB_NAME="$1"               # First argument: job identifier/name
JOB_EXECUTABLE_PATH="$2"    # Second argument: path to the job executable (e.g., sbatch script)

# === Environment Setup for Cron ===
# Source global environment if available (loads module command etc.)
if [ -f /etc/profile ]; then
    source /etc/profile
fi

# Optionally adjust PATH (if needed for `sbatch` or `module`)
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH

# === Validation ===
if [[ -z "$JOB_NAME" || -z "$JOB_EXECUTABLE_PATH" ]]; then
    echo "Usage: $0 <job_name> <job_executable_path>"
    exit 1
fi

if [[ ! -f "$JOB_EXECUTABLE_PATH" ]]; then
    echo "Error: Job executable path '$JOB_EXECUTABLE_PATH' does not exist."
    exit 2
fi

# === Check if job is already running or queued ===
IS_RUNNING=$(squeue -p orchid --format "%.300j" --me | grep -w "$JOB_NAME")

if [[ -z "$IS_RUNNING" ]]; then
    echo "$(date): No job named '$JOB_NAME' found in queue. Submitting job..."
    sbatch "$JOB_EXECUTABLE_PATH"
fi
# else
#     echo "$(date): Job '$JOB_NAME' is already running or queued. Skipping submission."
