#!/bin/bash

# Example usage:
# sbatch run_pre_training_job.sh

#SBATCH --job-name="pre_training_vanilla_t5_job"

#SBATCH -N 1                         # number of minimum nodes
#SBATCH --gres=gpu:5                 # Request n gpus
#SBATCH -w plato2
#SBATCH -p public
#SBATCH -A cs

#SBATCH -o eval_runs/slurm_%N_%j_out_PMI_T5_test.txt       # stdout goes here
#SBATCH -e eval_runs/slurm_%N_%j_err_PMI_T5_test.txt       # stderr goes here
#SBATCH --mail-type=fail         # send email if job fails

#SBATCH --mail-user=ofek.glick@campus.technion.ac.il
# To send you an email on failure, add 'SBATCH --mail-user=<your_mail>'

nvidia-smi
deepspeed pre_train_t5.py \
--model_name_or_path google/t5-v1_1-base     \
--tokenizer_name google/t5-v1_1-base     \
--pre_training_dataset_paths "wikipedia,bookcorpus" \
--pre_training_dataset_names "20220301.en,"    \
--input_seq_length 512    \
--target_seq_length 512    \
--mlm_probability 0.5    \
--mean_noise_span_length 3.0  \
--output_dir ./pre_training_outputs   \
--logging_dir ./pre_training_logs    \
--logging_steps 50    \
--eval_steps 500     \
--save_steps 500 \
--num_train_epochs 3 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4