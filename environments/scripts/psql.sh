#!/bin/bash
USER_NAME=$(id -un)
source .env

read -p "Which database in PostgreSQL container do you access? (mlflow or prefect or optuna): " TARGET_SERVER
# echo $TARGET_SERVER

if [ "$TARGET_SERVER" = "mlflow" ]; then
    echo "database: mlflow"
    psql -h $CONTAINER_NAME"_postgres" -p 5432 -U $USER_NAME -d mlflow

elif [ "$TARGET_SERVER" = "optuna" ]; then
    echo "container: optuna"
    psql -h $CONTAINER_NAME"_postgres" -p 5432 -U $USER_NAME -d optuna

elif [ "$TARGET_SERVER" = "prefect" ]; then
    echo "container: prefect"
    psql -h $CONTAINER_NAME"_postgres" -p 5432 -U $USER_NAME -d prefect
fi
