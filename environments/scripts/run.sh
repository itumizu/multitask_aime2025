#!/bin/bash
source .env

ARR=(${IMAGE_TAG//:/ })
IMAGE_TAG_POSTGERS=${ARR[0]}"_postgres:"${ARR[1]}
NETWORK_NAME=$CONTAINER_NAME"_network"
USER_NAME=$(id -un)
POSTGRES_DB_URI="postgresql+psycopg2://"$USER_NAME":"$POSTGRES_PASSWORD"@"$CONTAINER_NAME"_postgres:5432"

docker network create -d bridge $NETWORK_NAME

docker run  -it \
            -d \
            --rm \
            --init \
            --shm-size=350.0gb \
            -h $HOST_NAME \
            -e POSTGRES_USER=$(id -un) \
            -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
            -e POSTGRES_DB="mlflow" \
            -v $PWD/environments/db_server/scripts_sql:/docker-entrypoint-initdb.d \
            -v $OUTPUT_PATH_DEVICE/logs/:$PWD/logs/ \
            -v ./postgres/:/var/lib/postgresql/data \
            --expose 5432 \
            --name $CONTAINER_NAME"_postgres" \
            --network=$NETWORK_NAME \
            $IMAGE_TAG_POSTGERS

docker run --gpus all \
           -it \
           -d \
           --rm \
           --init \
           -h $HOST_NAME \
           -v $PWD:$PWD \
           -v $PWD/environments/etc/ssh:/etc/ssh \
           -v $PWD/environments/etc/home/.ssh/:/home/$(id -un)/.ssh \
           -v $OUTPUT_PATH_DEVICE/logs/:$PWD/logs/ \
           -v $OUTPUT_PATH_DEVICE/mlruns/:$PWD/mlruns/ \
           -v $OUTPUT_PATH_DEVICE_OPTION/results/:$PWD/results/ \
           -p 127.0.0.1:$PORT_SSH_MLFLOWSERVER:22 \
           --expose 5000 \
           --name $CONTAINER_NAME"_mlflow_server" \
           --network $NETWORK_NAME \
           $IMAGE_TAG

docker exec -it -d $CONTAINER_NAME"_mlflow_server" mlflow server --backend-store-uri ${POSTGRES_DB_URI}/mlflow --default-artifact-root ${PWD}/mlruns/ --host 0.0.0.0 --port 5000 --workers 4 --serve-artifacts

docker run --gpus all \
           -it \
           -d \
           --rm \
           --init \
           --shm-size=350.0gb \
           -h $HOST_NAME \
           -v $PWD:$PWD \
           -v $PWD/environments/etc/ssh:/etc/ssh \
           -v $PWD/environments/etc/home/.ssh/:/home/$(id -un)/.ssh \
           -v $OUTPUT_PATH_DEVICE/logs/:$PWD/logs/ \
           -v $OUTPUT_PATH_DEVICE/mlruns/:$PWD/mlruns/ \
           -v $OUTPUT_PATH_DEVICE_OPTION/results/:$PWD/results/ \
           -v $OUTPUT_PATH_DEVICE_OPTION/:$OUTPUT_PATH_DEVICE_OPTION \
           -p 127.0.0.1:$PORT_SSH:22 \
           --name $CONTAINER_NAME \
           --network $NETWORK_NAME \
           $IMAGE_TAG

docker exec -it --user $USER_NAME -d $CONTAINER_NAME ssh -f -N -L 5000:127.0.0.1:5000 -4 $CONTAINER_NAME"_mlflow_server" -i $PWD/environments/etc/home/.ssh/id_ed25519_container
