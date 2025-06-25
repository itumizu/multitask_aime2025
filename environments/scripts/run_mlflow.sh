#!/bin/bash
source .env

ssh -L 5000:localhost:5000 -4 $CONTAINER_NAME"_mlflow_server"