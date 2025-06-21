import os
import time
import uuid
import argparse
import traceback
import warnings
import subprocess
from pathlib import Path

import mlflow
import lightning.pytorch as pl
from ulid import ULID
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.experiments.run_experiment import runExperiment
from src.experiments.run_optuna import experimentCrossValidationWithOptuna
from src.utils.set_default_root_dir import setDefaltRootDir
from src.utils.run_prefect import runExperimentByPrefect
from src.utils.summarize import summarizeResults, IsFinisehedFoldRuns, compareExperimentResults
from src.utils.check_process import checkProcess

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def main(
    config_path: Path, gpus: list, fold_test_num: int, summarize=False, ulid_value: str = "", is_debug: bool = False, no_multiprocessing: bool = False
) -> None:
    config = OmegaConf.load(config_path)
    pl.seed_everything(seed=config.args.seed, workers=True)
    mlflow.set_tracking_uri(f'http://{os.environ["CONTAINER_NAME"] + "_mlflow_server:5000"}')

    experiment = mlflow.set_experiment(experiment_name=config.args.experiment_name)
    run_name = f"{config_path.parent.stem}/{config_path.stem}"
    config.args.ulid = ulid_value

    if no_multiprocessing:
        config.args.advance_setting.use_multiprocessing = False

    mean_value_dict: dict

    if config.args.optuna.use_optuna:
        try:
            mean_value_dict = experimentCrossValidationWithOptuna(
                config_path=config_path,
                config=config,
                gpus=gpus,
                run_name=run_name,
                experiment=experiment,
                fold_test_num=fold_test_num,
            )

        except:
            try:
                error_message = traceback.format_exc()
                print(error_message)

            except:
                pass

            raise

    else:
        default_root_dir = setDefaltRootDir(config=config)
        print(f"gpus: {gpus}")

        if len(gpus) > 1:
            import ray

            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpus[n % len(gpus)]) for n in range(0, 5)])
            ray.init(num_gpus=5)

        # elif len(gpus) == 1:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])

        try:
            log_dict = {"dictionary": config, "artifact_file": "config.yaml"}
            tags_dict = {"fold_test": fold_test_num, "ulid": ulid_value}

            mean_value_dict, output_dir = runExperiment(
                config=config,
                gpus=gpus,
                fold_test_num=fold_test_num,
                default_root_dir=default_root_dir,
                experiment_id=experiment.experiment_id,
                run_name=run_name,
                log_dict=log_dict,
                tags_dict=tags_dict,
                is_best=True,
                dir_name="",
            )

        except:
            try:
                error_message = traceback.format_exc()
                print(error_message)

            except:
                pass

            raise

    try:
        print("*****")
        for name, value in mean_value_dict.items():
            print(f"{name.replace('_', ' ')}: {value}")
        print("*****\n")

    except:
        pass

    print("Finished")

    if summarize:
        if not IsFinisehedFoldRuns(config_path=config_path, config=config, run_name=run_name, ulid_value=ulid_value):
            exit()

        try:
            _ = summarizeResults(config_path=config_path, config=config, run_name=run_name, fold_test_lengths=5)

        except:
            error_message = traceback.format_exc()
            print(error_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="", description="", epilog="")

    parser.add_argument(
        "--config_path",
        required=True,
        help="Path to the YAML file that specifies the experimental settings",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=[],
        help="IDs of the GPUs to use",
    )
    parser.add_argument(
        "--fold_test_num",
        type=int,
        default=None,
        help="Fold number to use for testing",
    )
    parser.add_argument(
        "--use_prefect",
        action="store_true",
        help="Register and run the job in Prefect",
    )
    parser.add_argument(
        "--use_high_priority",
        action="store_true",
        help="Register the job with high priority when running via Prefect",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="After execution, aggregate the 5-fold results and add them to Notion",
    )
    parser.add_argument(
        "--summarize_only",
        action="store_true",
        help="Aggregate the 5-fold results at the specified path and add them to Notion",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Refer to the specified YAML file and add the aggregated results to Notion",
    )
    parser.add_argument(
        "--ulid",
        type=str,
        default="",
        help="ULID value to distinguish the experiment",
    )
    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--no_multiprocessing", action="store_true")
    parser.add_argument("--output_path", type=str, default="", help="Output path")


    args = parser.parse_args()

    config_path = Path(args.config_path)
    gpus = args.gpus
    fold_test_num = args.fold_test_num
    use_prefect = args.use_prefect
    use_high_priority = args.use_high_priority
    summarize = args.summarize
    compare = args.compare
    summarize_only = args.summarize_only
    ulid_value = args.ulid
    is_debug = args.is_debug
    no_multiprocessing = args.no_multiprocessing
    output_path = args.output_path

    config = OmegaConf.load(config_path)

    if "conditions" in config:
        compare = True

    if fold_test_num is None and not summarize_only and not compare:
        fold_test_num_list = [num for num in range(0, 5)]
        use_prefect = True
        summarize = True

        ulid_value = str(ULID())

        # MPS
        if not checkProcess("nvidia-cuda-mps-control"):
            mps_shell_scripts_path = os.environ["MPS_SHELL_PATH"]
            subprocess.run([mps_shell_scripts_path], shell=True)

        if len(gpus) > 0:
            gpus_list = []

            if config.args.advance_setting.use_multiprocessing and not no_multiprocessing:
                for fold_test_num in fold_test_num_list:
                    gpus_list.append([str(gpus[n % len(gpus)]) for n in range(0, 5)])
                    # gpus_list.append([str(gpus[fold_test_num % len(gpus)]) for n in range(0, 5)])

                    if config.args.advance_setting.get("is_calculating_shap", False):
                        if gpus[0] == 1:
                            gpus = gpus[2:] + gpus[0:2]

                        else:
                            gpus = gpus[1:] + [gpus[0]]

                    else:
                        gpus = gpus[1:] + [gpus[0]]

            else:
                if len(gpus) == 1:
                    for fold_test_num in fold_test_num_list:
                        gpus_list.append([str(gpus[0])])

                else:
                    for fold_test_num in fold_test_num_list:
                        gpus_list.append([str(gpus[fold_test_num % len(gpus)])])

        else:
            gpus_list = [[]] * len(fold_test_num_list)

    elif not summarize_only and not compare:
        fold_test_num_list = [fold_test_num]
        gpus_list = [gpus]

        if config.args.advance_setting.use_multiprocessing and not no_multiprocessing and (len(gpus) > 0 and len(gpus) < 5):
            # gpus = [str(gpus[fold_test_num % len(gpus)]) for n in range(0, 5)]
            gpus = [str(gpus[n % len(gpus)]) for n in range(0, 5)]

    else:
        pass

    if use_prefect:
        script_file_dir = os.path.dirname(os.path.abspath(__file__))
        script_file_name = os.path.basename(__file__)

        if not summarize_only and not compare:
            for fold_test_num, gpus in zip(fold_test_num_list, gpus_list):
                print("use:", gpus)

                runExperimentByPrefect(
                    config_path=config_path,
                    gpus=gpus,
                    fold_test_num=fold_test_num,
                    script_file_dir=script_file_dir,
                    script_file_name=script_file_name,
                    use_high_priority=use_high_priority,
                    summarize=summarize,
                    compare=compare,
                    ulid_value=ulid_value,
                )

                time.sleep(5)

        else:
            runExperimentByPrefect(
                config_path=config_path,
                gpus=gpus,
                fold_test_num=fold_test_num,
                script_file_dir=script_file_dir,
                script_file_name=script_file_name,
                use_high_priority=use_high_priority,
                summarize=summarize,
                compare=compare,
                ulid_value=ulid_value,
                output_path=output_path,
            )

            time.sleep(5)

    print(f"Run completed: {config_path.parent.stem}/{config_path.stem}")
    exit()

    if summarize_only or compare:
        if compare == summarize_only:
            raise ValueError("You cannot specify both options at the same time")

        if summarize_only:
            config = OmegaConf.load(config_path)
            run_name = f"{config_path.parent.stem}/{config_path.stem}"
            summarizeResults(
                config=config,
                config_path=config_path,
                run_name=run_name,
                fold_test_lengths=5,
            )

        elif compare:
            use_fast = False

            config = OmegaConf.load(config_path)
            config_name_dict = {}
            config_path_list = []

            if isinstance(config.conditions, ListConfig):
                config_path_list = [Path(path) for path in config.conditions]

            elif isinstance(config.conditions, DictConfig):
                for path, name in config.conditions.items():
                    path = str(path)
                    config_path_list.append(Path(path))
                    config_name_dict[Path(path)] = name

            compareExperimentResults(
                config_path,
                config,
                config_path_list,
                config_name_dict,
                use_fast,
                output_path=output_path,
            )

        exit()

    if not ulid_value:
        ulid_value = str(ULID())

    if len(gpus) > 1:
        if (
            not config.args.advance_setting.use_multiprocessing
            and not config.args.advance_setting.get("use_fsdp", False)
        ):
            raise ValueError(
                "Multiple GPUs are configured, but `use_multiprocessing` is not set to True."
            )

    main(
        config_path=config_path,
        gpus=gpus,
        fold_test_num=fold_test_num,
        summarize=summarize,
        ulid_value=ulid_value,
        is_debug=is_debug,
        no_multiprocessing=no_multiprocessing,
    )
