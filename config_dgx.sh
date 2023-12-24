# Accept a parameter corresponding to the desired name of the container.
# Default is "nanoT5"
export CONTAINER_NAME=$1
export BASE_IMAGE_NAME=$2
export BASE_IMAGE_PATH=$3


# Assign default values to the parameters if they are not provided.
if [ -z "$BASE_IMAGE_PATH" ] || [ -z "$BASE_IMAGE_NAME" ]; then
    echo "Either the base image path or the base image name was not provided."
    echo "Using default base image path and name."
    export BASE_IMAGE_NAME="nvidia+pytorch+23.12-py3.sqsh"
    export BASE_IMAGE_PATH="docker://nvcr.io#nvidia/pytorch:23.12-py3"
fi

if [ -z "$CONTAINER_NAME" ]; then
    echo "No container name provided. Using default: $CONTAINER_NAME"
    export CONTAINER_NAME="nanoT5"
fi

# Set the following environment variables:
# (1) ENROOT_MOUNT_HOME=yes - Mount the user's home directory in the container
export ENROOT_MOUNT_HOME=yes

# Create a pytorch container via enroot on an HPC cluster
# Creates a file called "nvidia+pytorch+23.12-py3.sqsh" in the current directory
# If the file already exists, it will not be recreated, and instead print a message.
if [ ! -f "$BASE_IMAGE_NAME" ]; then
    echo "Creating container..."
    enroot "import $BASE_IMAGE_PATH"

    # Set the TMPDIR environment variable to $cwd/tmp
    expert TMPDIR="$PWD"/tmp

else
    echo "Container already exists. Skipping creation."
fi

# Create a container with the provided name.
echo "Creating container with name: $CONTAINER_NAME"
enroot create --name $CONTAINER_NAME $BASE_IMAGE_NAME



# Install python packages in the container.
#enroot --start --rw run -t $CONTAINER_NAME -- bash -c "apt-get update && apt-get install -y python3-pip && pip3 install torch transformers sentencepiece hydra-core omegaconf"


# Modify and save the container with the provided name.
# Update the container with the latest apt-get packages and install pip3.
# Install the following python packages:
# (1) torch, (2) torchvision, (3) transformers, (4) sentencepiece, (5) hydra and (6) omegaconf

# Previous version of the container
#srun -p mig -G 1 \
#--container-image=./nvidia+pytorch+23.12-py3.sqsh \
#--container-save=./ml_training.sqsh \
#--pty bash


#srun -p mig -G 1 \
#--container-image="./nvidia+pytorch+23.12-py3.sqsh" \
#--container-name="./$CONTAINER_NAME" \
#--pty bash -c "apt-get update && apt-get install -y python3-pip && pip3 install torch transformers sentencepiece hydra-core omegaconf"

# Remove the container once it is no longer needed.
enroot remove $CONTAINER_NAME