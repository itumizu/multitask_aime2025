#!/bin/bash
source .env
USER_NAME=$(id -un)

export PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://${USER_NAME}:${POSTGRES_PASSWORD}@${CONTAINER_NAME}_postgres:5432/prefect"
echo $PREFECT_API_DATABASE_CONNECTION_URL

prefect server start