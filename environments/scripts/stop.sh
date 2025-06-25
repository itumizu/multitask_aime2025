#!/bin/bash
source .env

docker stop $CONTAINER_NAME
docker stop $CONTAINER_NAME"_postgres"
docker stop $CONTAINER_NAME"_mlflow_server"