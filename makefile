.PHONY: prefect prefect_workpool

run:
	./environments/scripts/run.sh

stop:
	./environments/scripts/stop.sh

build:
	./environments/scripts/build.sh

ssh:
	./environments/scripts/ssh.sh

env:
	./environments/scripts/env.sh

mlflow:
	./environments/scripts/run_mlflow.sh

psql:
	./environments/scripts/psql.sh
	
prefect:
	tmux new -s "prefect_server" -d ./environments/scripts/run_prefect.sh

prefect_workpool:
	./environments/scripts/run_workpool.sh