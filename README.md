# Multi-Task Learning on Tabular Health Checkup Data for Prediction of Lifestyle-Related Diseases
## About this repository
This is the official implement of 

This environment consists of three containers: main, PostgreSQL, and MLflow.  

The main container is in which you execute experiments and analyze data.  
The MLflow container is the endpoint to which you sent the experiment results and the server running the MLflow Web UI.  
The PostgreSQL container is the database used by MLflow and Prefect. You can also use this container for other purposes.  

## Base image
```
nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
```

## Default python version
```
3.11.5
```
If you want to use other version, please change the python version in Dockerfile.

## How to set up the environment ?

### 1. Make .env file
We provide a example of `.env`, so you copy this file as template, please.
```
cp .env.example .env
```

### 2. Write your configure to `.env`
Plaase change it if necessary.  
Details of `.env` are given below.  
```
IMAGE_TAG="{user_name}/{project_name}:{build date yyyymmdd}" <- Image tag ex) "user_name/project_name:yyyymmdd"
CONTAINER_NAME=""                                            <- container name to distinguish it from other users' containers.
POSTGRES_PASSWORD=""                                         <- Password of PostgreSQL server. This phrase automatically sets the PostgreSQL password when you first start the container. 
                                                                The PostgreSQL username will be the username of the user who built the image.
PORT_SSH=""                                                  <- SSH port of main container. Please select port number from dynamic, private or ephemeral ports.
PORT_SSH_MLFLOWSERVER=""                                     <- SSH port of MLFlow container. Please select port number from dynamic, private or ephemeral ports.
HOST_NAME="localhost"                                        <- No need to change it.
CUDA_LAUNCH_BLOCKING=1                                       <- This variable and its value allow for a detailed display of CUDA errors.
```

### 3. Add the APT packages and Python libraries.
If you want to add an apt package, you have to write the package name to Dockerfile directly.  
`./enviroments/Dockerfile`

```
RUN apt install -y build-essential ca-certificates locales-all \
    libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev \
    liblzma-dev zlib1g zlib1g-dev uuid-dev libffi-dev libdb-dev bzip2 checkinstall libreadline-gplv2-dev tk-dev \
    software-properties-common git emacs vim curl unzip htop openssh-server wget procps sudo nodejs tmux postgresql-client <- Please add the package behind here.
```

For Python packages, please add the library name and version to `requirements.txt`.
`./environments/python/requirements.txt`
```
library==ver.sion
```
### 4. Build docker image.
Please execute the below "make" command to build your image.

```
make build
```  
Follow the script to set the key pair and password to log in to the container.  
The key pair will be created in the following folder: `./environments/etc/home/.ssh`.  
If you save this private key on your PC, you can use an SSH connection with exchange key encryption.  
If you want to connect without exchanging key ciphers, you can use the password you set here.

### 5. Run docker container.
```
make run
```

### 6. Connect using SSH
A user with the same username as the user who built the docker container is prepared inside the container.  
Please connect to the container using your username.  
```
make ssh 
```
or 
```
ssh {user_name}@127.0.0.1 -p {PORT_SSH}
```

This container cannot be directly connected from the outside.  
Therefore, you have to set up a connection to the container via the host.  

## How to connect prefect ans mlflow web UI ?
The default ports for Prefect and MLflow are 4200 and 5000.  
Please connect via `http://127.0.0.1:4200` and `http://127.0.0.1:5000` using SSH port forwarding.  

## How to reproduce the results
We obtained a health checkup dataset actually at a healthcare center. Therefore, we cannot be made public.  
The code used in the experiment runs as follows:  

1. Run the prefect server and prefect workpool by executing following commands.
    ```
    make prefect
    make prefect_workpool
    ```

2. Run the following commands.  
    ```
    cd src/multi_task/
    python main.py --config_path args/conditions/*.yaml --gpus 0 --use_prefect
    ```

## How to preprocess health checkup dataset
We used the following scripts for preprocessing the dataset.

```
./src/preprocess/main.py
```