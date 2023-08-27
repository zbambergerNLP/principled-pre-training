#!/bin/bash

# Example usage:
# sbatch run_pretraining.bash

#SBATCH --job-name="eval_job"

#SBATCH -N 1                         # number of minimum nodes
#SBATCH --gres=gpu:1                 # Request n gpus

#SBATCH -p public
#SBATCH -A cs

#SBATCH -o eval_runs/slurm_%N_%j_out.txt       # stdout goes here
#SBATCH -e eval_runs/slurm_%N_%j_err.txt       # stderr goes here
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ofek.glick@campus.technion.ac.il


nvidia-smi
CUDA_VISIBLE_DEVICES=0 wandb agent --count 1 'ofek-gluck/T5 Evaluation/jhovoaxh'



