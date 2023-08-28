# Training and Evaluation of Encoder-Decoder Language Models

## Introduction

This repository provides a complete toolkit for training and fine-tuning T5 models using span masking 
(an extension of Masked language Modeling, a.k.a., MLM, as described in [BERT](https://arxiv.org/abs/1810.04805)).


Leveraging libraries like Hugging Face's [`transformers`](https://huggingface.co/docs/transformers/index) and [`accelerate`](https://huggingface.co/docs/accelerate/index)
, Microsoft's [`DeepSpeed`](https://www.deepspeed.ai/), and [`wandb`](https://wandb.ai/site) it offers a robust and user-friendly 
platform for experimentation with state-of-the-art models.

## Table of Contents

1. [Distributed Training](#distributed-training)
2. [Tips for environment setup (within PyCharm)](#tips-for-environment-setup-within-pycharm)
3. [Setup and Installation](#setup-and-installation)
4. [Distributed Training](#distributed-training)
      1. [Distributed Training with Accelerate](#distributed-training-with-accelerate)
      2. [Distributed Training with DeepSpeed](#distributed-training-with-deepspeed)
5. [Pipelines](#pipelines)
   1. [Step-by-Step Pre-Training Process](#step-by-step-pre-training-process)
   2. [Step-by-Step Fine-Tuning Process](#step-by-step-fine-tuning-process)
   3. [Sweeps](#sweeps)
6. [Design Decisions](#design-decisions)
7. [Expanding Fine-Tuning Capabilities](#expanding-fine-tuning-capabilities)
8. [Troubleshooting and FAQs](#troubleshooting-and-faqs) 
9. [Acknowledgments and References](#acknowledgments-and-references)

# Training and Evaluation of Encoder-Decoder Language Models

## Tips for environment setup (within PyCharm)

* Create a new project in PyCharm, and use the Conda interpreter with Python 3.10. See a useful guide linked [here](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html).
* Use PyCharm's VCS functionality to clone this repository into your project as described [here](https://www.jetbrains.com/help/pycharm/manage-projects-hosted-on-github.html).
* Install the required packages (see [Setup and Installation](#setup-and-installation) below).
* Within your distributed computing server, set up a new conda virtual environment with Python 3.10 as you did locally.
* Set up a deployment configuration in PyCharm such that local changes are automatically uploaded to the server.
  You can find a useful guide [here](https://www.jetbrains.com/help/pycharm/tutorial-deployment-in-product.html#summary)
  * It is recommended to work with [GitHub Co-Pilot](https://docs.github.com/en/copilot/getting-started-with-github-copilot)
  for continued development locally. This is free for students (as descirbed on [this page](https://docs.github.com/en/copilot/quickstart))!
* On the remote server, install the required packages (as you did above).
* Run `accelerate config` to set up the distributed training configuration as described in [distributed training](#distributed-training) below.
* Run `wandb login` to set up the Weights and Biases integration.
* Run a training script:
  * Run `accelerate launch fine_tune_t5.py` to start fine-tuning with accelerate, or run `accelerate launch pre_train_t5.py` to start pre-training with accelerate.
  * If you want to use DeepSpeed instead, run `deepspeed <script_name>` (Make sure you specified the correct settings in the configuration step above). You will need to point to the correct deepspeed configuration file (`zer0_stage2_config.json`).

**Note:** At this point we only support [ZeRO-2](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/) with DeepSpeed. 
We are working on adding support for [ZeRO-3](https://deepspeed.readthedocs.io/en/latest/zero3.html) in the near future.
For more information about these tools for distributed training, see the [ZeRO paper](https://arxiv.org/abs/1910.02054).

## Setup and Installation

First, install Anaconda or Miniconda. Then, create a new conda environment and install the required packages
with the following commands (see reference [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)):

```bash
conda env create -f conda_environment_slurm.yml
conda activate ml_training
```

## Distributed Training

### Distributed Training with Accelerate

We currently support only single-node multi-GPU training. To train on a single node with 4 GPUs, run:
```accelerate config```

When prompted, select the following options:
```
In which compute environment are you running? <This machine>                                                                                                                                                                                                
Which type of machine are you using? <multi-GPU>  
How many different machines will you use (use more than 1 for multi-node distributed training)? <1>
Do you wish to optimize your script with torch dynamo? <no>
Do you want to use DeepSpeed? <no>
Do you want to use FullyShardedDataParallel? <no> 
Do you want to use Megatron-LM? <no> 
How many GPUs should be used for distributed training? <4>
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all] <enter>
Do you wiush to use FP16 or BF16? <FP16>
```

Next, make sure you are logged into wandb so that you can track your training runs (if prompted, follow the 
instructions to create a free account):
```wandb login```

Once you've configured the accelerator, and set up wandb, you can run a training script such as:
```accelerate launch fine_tune_t5.py```


### Distributed Training with DeepSpeed

To train with DeepSpeed, you must first install it:
```pip install deepspeed```

Then, you can configure the accelerator with:
```accelerate config```

When prompted, select the following options:
```
In which compute environment are you running? <This machine>                                                                                                                                                                                                
Which type of machine are you using? <multi-GPU>                                                                                                                                                                                                   
How many different machines will you use (use more than 1 for multi-node distributed training)? <1>
Do you wish to optimize your script with torch dynamo? <no>                                                                                                                                       
Do you want to use DeepSpeed? [yes/NO]: <yes>                                                                                                                                                                 
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: <yes>                                                                                                                                     
Please enter the path to the json DeepSpeed config file: <zero_stage2_config.json>
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: <no>
How many GPU(s) should be used for distributed training? [1]: <4>
``` 

Next, make sure you are logged into wandb so that you can track your training runs (if prompted, follow the
instructions to create a free account):
```wandb login```

Once you've configured the accelerator, and set up wandb, you can run a training script such as:
```deepspeed fine_tune_t5.py```

**Note:** Zero3 is not yet supported.

## Pipelines

### Step-by-Step Pre-Training Process

1. **Flag Parsing**: Parse the necessary flags using `HfArgumentParser`.
2. **Accelerator Initialization**: Initialize `accelerate.Accelerator`.
3. **Model and Tokenizer Loading**: Load the T5 model and tokenizer.
4. **W&B Initialization**: Set up Weights & Biases for experiment tracking.
5. **Data Loading and Preprocessing**: Load datasets and tokenize.
6. **Data Collator Initialization**: Utilize `T5DataCollator` with span corruption.
7. **Metrics Configuration**: Set up evaluation metrics.
8. **Trainer Initialization**: Initialize `transformers.Seq2SeqTrainer`.
9. **Training Execution**: Start pre-training.


### Step-by-Step Fine-Tuning Process

1. **Flag Parsing**: Parse the necessary flags.
2. **Seed Setting**: Set random seed.
3. **Accelerator Initialization**: Initialize `accelerate.Accelerator`.
4. **Model and Tokenizer Loading**: Load the T5 model and tokenizer.
5. **W&B Initialization**: Set up Weights & Biases.
6. **Data Loading and Preprocessing**: Load dataset and encode.
7. **Metrics Configuration**: Set up evaluation metrics.
8. **Trainer Initialization**: Initialize `transformers.Seq2SeqTrainer`.
9. **Training Execution**: Start fine-tuning.

### Sweeps

We use [Weights & Biases sweeps](https://docs.wandb.ai/guides/sweeps) to run hyperparameter optimization.

To run a sweep, first set up the sweep configuration file (`sweep_config.yaml`) with the desired hyperparameters.
Then, run the sweep with `wandb sweep sweep_config.yaml`.
Finally, run the sweep agent with `wandb agent <sweep_id>`.

We have provided sweep configurations for fine-tuning T5 on GLUE tasks (see the `sweeps` directory). 

If you are running sweeps on a remote server, you can run `wandb sweep <sweep_config_file>` without `srun` or `sbatch`.
However, you will need to run `wandb agent <sweep_id>` with `srun` or `sbatch` to ensure that the sweep agent is running
on the correct machine.

See the following example sequence of commands:

```bash
wandb sweep sweeps/glue_sst2_sweep.yaml
srun --account <account_name> --partition <partition_name> --gres=gpu:<num_gpus> wandb agent <sweep_id>`
```

Where the sweep ID is outputted by the `wandb sweep` command. Make sure that the number of GPUs you request is the same
as the number of GPUs you specified when running `accelerate config`.

See an example yaml file for a sweep configuration below:

```yaml
program: fine_tune_t5.py
project: "T5 Evaluation"
name: "T5 Evaluation -- GLUE: SST-2"
method: bayes
metric:
  name: eval/accuracy
  goal: maximize

parameters:
  learning_rate:
    distribution: uniform
    min: 1e-5
    max: 1e-3

  lr_scheduler_type:
    values: [
      "constant_with_warmup",
      "linear",
      "cosine",
    ]
  benchmark:
    value: 'glue'
  dataset_name:
    value: "sst2"

command:
  - accelerate
  - launch
  - ${program}
  - ${args}
```

**Note:** We use the `accelerate` launcher to run the training script. This is necessary for distributed training.

**Note:** We focus specifically on the learning rate and scheduler type hyperparameters since these are the most important for
fine-tuning T5. However, you can add more hyperparameters as needed.

## Design Decisions


- **Weights & Biases (W&B)**: For experiment tracking and logging.
- **Accelerate**: For distributed training.
- **DeepSpeed**: An alternative for distributed training. Flexibility to support larger models.
- **Absil Parameterized Unit Tests**: For efficient and readable testing.
- **Lambda Functions**: For dynamic function instantiation (metric computation and tokenizer functions). 
   Permissible for the `map` function for dataset tokenization, and the metrics computation hook executed by HuggingFace trainers.


## Expanding Fine-Tuning Capabilities

The fine-tuning process in this repository is designed to be extensible, allowing users to easily add more datasets 
for task-specific training. Here's how you can expand the fine-tuning capabilities to more datasets:

### Constants Configuration (`constants.py`)

The `constants.py` file includes a dictionary `DATASET_VALS` that defines the configurations for various datasets,
such as prefixes, column names, metrics, and labels. You can add new datasets by extending this dictionary with the required details.

### Tokenization and Metric Computation

The fine-tuning script (`fine_tune_t5.py`) leverages the constants to dynamically instantiate functions for tokenization and metric computation using lambda functions. 
This approach enhances flexibility and modularity, allowing seamless integration of new datasets.

### Example Structure for a New Dataset

```python
DATASET_VALS = {
  'DATASET_NAME': {
      'task_name': {
          'prefix': 'your_prefix',
          'text_column_name': 'text_column',
          'label_column_name': 'label_column',
          'metric_to_optimize': 'chosen_metric',
          'metric_names': ['metric1', 'metric2'],
          'labels': {
              0: 'label0',
              1: 'label1',
          },
      },
  },
}

```

## Troubleshooting and FAQs
This section provides guidance on common issues and frequently asked questions:

* **Q**: How can I set up distributed training?

  **A**: Follow the distributed training guide provided in the [Distributed Training section](#distributed-training) above.

* **Q**: Can I use this repository for other benchmarks and tasks? 

  **A**: Yes, the repository is designed to be extensible and can be adapted for various NLP benchmarks and tasks.

For more specific inquiries or troubleshooting, please feel free to open an issue on GitHub or contact the maintainers.


## Acknowledgments and References


- **Original T5 Paper**: [link](https://arxiv.org/abs/1910.10683)
- **Contributors and Collaborators**: Zach Bamberger, Ofek Glick, Jonny Gal
- **Supervision**: Yonatan Belinkov and Chaim Baskin.
