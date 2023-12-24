# Call run_pretraining.sh from nanoT5/run_job.sh

export NUM_GPUS=$1
expert NUM_CPUS=$2
export LOG_PATH=$3
export JOB_NAME=$4

# Set default number of GPUs if not provided
if [ -z "$NUM_GPUS" ]; then
    export NUM_GPUS=1
    echo "No number of GPUs provided. Using default: $NUM_GPUS"
fi

# Set default number of CPUs if not provided
if [ -z "$NUM_CPUS" ]; then
    export NUM_CPUS=1
    echo "No number of CPUs provided. Using default: $NUM_CPUS"
fi

# Set default log path if not provided
if [ -z "$LOG_PATH" ]; then
    export LOG_PATH="slurm_%A_%a_%N_out.txt"
    echo "No log path provided. Using default: $LOG_PATH"
fi

# Set default job name if not provided
if [ -z "$JOB_NAME" ]; then
    export JOB_NAME="nanoT5_job"
    echo "No job name provided. Using default: $JOB_NAME"
fi

sbacth --job-name="$JOB_NAME" \
--output=$LOG_PATH \
--nodes=1 \
--ntasks-per-node=1 \
--gres=gpu:$NUM_GPUS \
--cpus-per-task=$NUM_CPUS \
