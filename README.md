# Training and Evaluation of Encoder-Decoder Language Models

## Tips for environment setup (within PyCharm)

* Create a new project in PyCharm, and use the Conda interpreter with Python 3.10.
* Use PyCharm's VCS functionality to clone this repository into your project.
* Install the required packages (see below).
* Within your distributed computing server, set up a new conda virtual environment with Python 3.10 as you did locally.
* Set up a deployment configuration in PyCharm such that local changes are automatically uploaded to the server.
  * It is recommended to work with GitHub Co-Pilot for continued development locally. This is free for students!
* On the remote server, install the required packages (see below).
* Run `accelerate config` to set up the distributed training configuration as described below.
* Run `wandb login` to set up the Weights and Biases integration as described below.
* Run `accelerate launch fine_tune_t5.py` to start training.

## Setup and Installation

First, install Anaconda or Miniconda. Then, create a new conda environment and install the required packages:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge transformers
conda install -c conda-forge accelerate
conda install -c conda-forge sentencepiece
conda install -c anaconda scikit-learn
pip install wandb
```

### Distributed Training

We currently support only single-node multi-GPU training. To train on a single node with 8 GPUs, run:
```accelerate config```

When prompted, select the following options:
```
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

Once you've configured the accelerator, and set up wandb, you can run the training script with:
```accelerate launch fine_tune_t5.py```