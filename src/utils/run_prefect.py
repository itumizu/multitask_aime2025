import os
from pathlib import Path
from datetime import datetime
from typing import Any

from prefect import flow
from prefect.deployments.deployments import Deployment, run_deployment
from prefect_shell import ShellOperation
import ulid


@flow(log_prints=False, flow_run_name="run_{current_date}")
def runExperimentCommand(script_file_dir: Path, python_command: str, current_date: str):
    shell = ShellOperation(
        commands=[
            f"{python_command}",
        ],
        stream_output=False,
        working_dir=script_file_dir,
    )

    shell.run()


def runExperimentByPrefect(
    config_path: Path,
    gpus: list,
    fold_test_num: int,
    script_file_dir: str,
    script_file_name: str,
    use_high_priority: bool = False,
    summarize: bool = False,
    compare: bool = False,
    ulid_value: str = "",
    output_path: str = "",
):
    run_name = f"{config_path.parent.stem}_{config_path.stem}_fold_test_{fold_test_num}"
    script_file_path = Path(script_file_dir)
    current_day = datetime.now().strftime("%Y-%m-%d")
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    python_command = f"python3 {script_file_name} --config_path {str(config_path)}"

    if not compare:
        python_command += f" --fold_test_num {str(fold_test_num)}"

    if len(gpus) > 0:
        python_command += f" --gpus {' '.join(list(map(str, gpus)))}"

    if summarize:
        python_command += f" --summarize"

    if ulid_value:
        python_command += f" --ulid {ulid_value}"

    if compare:
        python_command = f"python3 {script_file_name} --config_path {str(config_path)} --compare"

    if output_path:
        python_command += f" --output_path {output_path}"

    log_dir = Path(
        os.environ["PYTHONPATH"],
        "logs",
        current_day,
        f"exp_{run_name}",
    )
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = Path(log_dir, f"{current_date}.log")
    python_command = python_command + f" > {str(log_file_path)} 2>&1"
    # python_command = f'tmux new-session -d -s exp_{run_name} "{python_command}" && tmux attach-session -t exp_{run_name}'

    deployment: Any = Deployment.build_from_flow(flow=runExperimentCommand, name=run_name, work_pool_name="experiment_multitask")
    deployment = deployment.apply()

    work_queue_name = None

    if use_high_priority:
        work_queue_name = "high_priority"

    run_deployment(
        name=deployment,
        timeout=0,
        parameters={"script_file_dir": script_file_path, "python_command": python_command, "current_date": current_date},
        work_queue_name=work_queue_name,
    )
