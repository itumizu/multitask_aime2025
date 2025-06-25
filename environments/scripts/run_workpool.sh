#!/bin/bash
source .env
USER_NAME=$(id -un)

export PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://${USER_NAME}:${POSTGRES_PASSWORD}@${CONTAINER_NAME}_postgres:5432/prefect"
WORKPOOL_NAME="experiment_multitask"

prefect work-pool create $WORKPOOL_NAME --type process
prefect work-pool update $WORKPOOL_NAME --concurrency-limit 20

prefect work-queue set-concurrency-limit default 1 --pool $WORKPOOL_NAME
prefect work-queue create high_priority --limit 10 --pool $WORKPOOL_NAME --priority 1

tmux new -s "prefect_workpool_"$WORKPOOL_NAME -d "prefect worker start --pool "$WORKPOOL_NAME" --work-queue default"
tmux new -s "prefect_workpool_"$WORKPOOL_NAME"_high_priority" -d "prefect worker start --pool "$WORKPOOL_NAME" --work-queue high_priority"
