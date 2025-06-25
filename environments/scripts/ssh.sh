#!/bin/bash

source .env
USER_NAME=$(id -un)

read -p "Which container do you access? default or mlflow: " TARGET_SERVER
# echo $TARGET_SERVER

if [ "$TARGET_SERVER" = "mlflow" ]; then
    echo "container: mlflow server"
    ssh ${USER_NAME}@localhost -p ${PORT_SSH_MLFLOWSERVER} -i ./environments/etc/home/.ssh/id_ed25519_container
else
    echo "container: default"
    ssh ${USER_NAME}@localhost -p ${PORT_SSH} -i ./environments/etc/home/.ssh/id_ed25519_container
fi