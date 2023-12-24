#!/bin/bash
#SBATCH --job-name="pre_training_job"

# %A is the job id (e.g. 12345), %a is the array id (e.g. 0), %N is the node name (e.g. node-123).
#SBATCH -o pre_training_runs/slurm_%A_%a_%N_out.txt       # stdout goes here
#SBATCH -e pre_training_runs/slurm_%A_%a_%N_err.txt       # stderr goes here

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:6                # number of GPUs per node
#SBATCH --cpus-per-task=80          # number of cores per tasks

#SBATCH -p nlp
#SBATCH -A nlp
#SBATCH -w nlp-ada-2

#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il

nvidia-smi

######################
### Set enviroment ###
######################
export GPUS_PER_NODE=6
######################

#export SCRIPT="./main.py"
#export SCRIPT=/accelerate/examples/complete_nlp_example.py
#export SCRIPT_ARGS=" \
#    --mixed_precision fp16 \
#    --output_dir /accelerate/examples/output \
#    --with_tracking \
#    "
#accelerate launch --num_processes $GPUS_PER_NODE $SCRIPT "$SCRIPT_ARGS"
#accelerate launch -m --num_processes $GPUS_PER_NODE $SCRIPT

accelerate launch -m \
--mixed_precision bf16 \
--num_cpu_threads_per_process 64 \
--num_processes 6 \
nanoT5.main \
optim.name=adamwscale \
optim.lr_scheduler=cosine \
model.compile=true \
num_gpus=6 \
num_cpus=64 \
data.num_workers=64

#
#
#python -m nanoT5.main \
#    optim.name={adafactor,adamwscale} \
#    optim.lr_scheduler={legacy,cosine}