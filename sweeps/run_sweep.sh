#!/bin/bash

# Example usage:
# sbatch run_pretraining.bash

#SBATCH --job-name="eval_job"

#SBATCH -N 1                         # number of minimum nodes
#SBATCH --gres=gpu:4                 # Request n gpus

#SBATCH -p nlp
#SBATCH -A nlp

#SBATCH -o eval_runs/slurm_%N_%j_out.txt       # stdout goes here
#SBATCH -e eval_runs/slurm_%N_%j_err.txt       # stderr goes here
#SBATCH --mail-type=fail         # send email if job fails

# To send you an email on failure, add 'SBATCH --mail-user=<your_mail>'

# Run the following command to start the sweep (allow for 20 runs at a time):
#nvidia-smi
#wandb agent --count 20 <your sweep ID>



