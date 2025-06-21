import traceback
from pathlib import Path

import torch
import lightning.pytorch as pl
from omegaconf import ListConfig, DictConfig

from src.utils.select_model import selectModel
from src.modules.module import HealthExperimentModule
from src.utils.neural_network.select_callbacks import selectCallbacks, ExperimentLoggingCallback
from src.utils.calc_shap_values import calculateSHAPValues


def trainNeuralNetworkModel(
    config: DictConfig | ListConfig,
    gpus: list,
    fold_test_num: int,
    fold_val_num: int,
    default_root_dir_fold: Path,
    return_with_model=False,
    return_validation_results_only=False,
) -> dict:
    accelerator = "cpu"
    devices: list | str = "auto"

    if len(gpus) > 0:
        accelerator = "gpu"
        devices = gpus

        float32_matmul_precision = config.args.advance_setting.get("float32_matmul_precision", "highest")
        torch.set_float32_matmul_precision(float32_matmul_precision)

    accumulate_grad_batches = config.args.dataset.get("accumulate_grad_batches", 1)
    precision = config.args.advance_setting.get("precision", "32-true")
    use_multiprocessing = config.args.advance_setting.get("use_multiprocessing", False)
    use_fsdp = config.args.advance_setting.get("use_fsdp", False)
    strategy = "ddp" if use_fsdp else "auto"

    model = selectModel(config=config, gpus=gpus, fold_test_num=fold_test_num, fold_val_num=fold_val_num)
    module = HealthExperimentModule(config=config, model=model, fold_test_num=fold_test_num, fold_val_num=fold_val_num, accelerator=accelerator, devices=devices)

    callbacks = selectCallbacks(config=config, use_multiprocessing=use_multiprocessing)
    callback_logger = ExperimentLoggingCallback(config=config)
    callbacks.append(callback_logger)

    trainer = pl.Trainer(
        deterministic="warn",
        accelerator=accelerator,
        devices=devices,
        default_root_dir=default_root_dir_fold,
        max_epochs=config.args.optimizer.max_epochs,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=not (use_multiprocessing and not config.args.advance_setting.get("show_progress", False)),
        precision=precision,
        strategy=strategy,
    )

    trainer.fit(model=module)

    results_dict = {}
    val_results_dict = {}
    test_results_dict = {}
    test_shap_dict = {}
    test_df_true_pred_dict = {}
    test_attention_dict = {}

    logging_dict = module.logging_dict
    ckpt_dict = {ckpt.monitor: {"best": ckpt.best_model_path, "last": ckpt.last_model_path} for ckpt in trainer.checkpoint_callbacks}
    monitor_key = config.args.callbacks.model_checkpoint.monitor

    if config.args.callbacks.model_checkpoint.get("evaluate_all_metrics", False):
        check_ckpt_dict = ckpt_dict
    else:
        check_ckpt_dict = {monitor_key: ckpt_dict[monitor_key]}

    if return_validation_results_only:
        ckpt_path = ckpt_dict[monitor_key]["best"]
        trainer.validate(model=module, ckpt_path=ckpt_path, verbose=False)
        val_results = module.results_dict["val"]
        val_results_dict.update({
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in val_results["metrics_dict"].items()
        })
    else:
        for monitor_name, ckpt_path_dict in check_ckpt_dict.items():
            ckpt_path = ckpt_path_dict["best"]
            monitor_prefix = f"_monitored_by_{monitor_name}" if monitor_name != monitor_key else ""

            model = selectModel(config=config, gpus=gpus, fold_test_num=fold_test_num, fold_val_num=fold_val_num)
            module = HealthExperimentModule(config=config, model=model, fold_test_num=fold_test_num, fold_val_num=fold_val_num, accelerator=accelerator, devices=devices)
            callbacks = selectCallbacks(config=config, use_multiprocessing=use_multiprocessing)
            callback_logger = ExperimentLoggingCallback(config=config)
            callbacks.append(callback_logger)

            trainer = pl.Trainer(
                deterministic="warn",
                accelerator=accelerator,
                devices=devices,
                default_root_dir=default_root_dir_fold,
                max_epochs=config.args.optimizer.max_epochs,
                callbacks=callbacks,
                accumulate_grad_batches=accumulate_grad_batches,
                enable_progress_bar=True,
                precision=precision,
            )

            trainer.validate(model=module, ckpt_path=ckpt_path, verbose=False)
            val_results = module.results_dict["val"]
            val_results_dict.update({
                f"{k}{monitor_prefix}": v.item() if isinstance(v, torch.Tensor) else v
                for k, v in val_results["metrics_dict"].items()
            })

            trainer.test(model=module, ckpt_path=ckpt_path, verbose=False)
            test_results = module.results_dict["test"]
            df_true_pred = test_results["df_true_pred"]

            if "attention" in test_results:
                test_attention_dict[f"attention{monitor_prefix}"] = test_results["attention"]

            try:
                shap_values_dict = calculateSHAPValues(
                    config=config,
                    model=model,
                    fold_test_num=fold_test_num,
                    fold_val_num=fold_val_num,
                    device=torch.device(f"cuda:{devices[0]}" if accelerator == "gpu" else "cpu"),
                )
            except:
                print(traceback.format_exc())
                shap_values_dict = {}

            df_true_pred.columns = [f"{col}{monitor_prefix}" for col in df_true_pred.columns]
            test_shap_dict[f"shap{monitor_prefix}"] = shap_values_dict
            test_df_true_pred_dict[f"df_true_pred{monitor_prefix}"] = df_true_pred

            test_results_dict.update({
                f"{k}{monitor_prefix}": v.item() if isinstance(v, torch.Tensor) else v
                for k, v in test_results["metrics_dict"].items()
            })

    results_dict["val_results"] = val_results_dict

    if not return_validation_results_only:
        results_dict["test_results"] = test_results_dict
        results_dict["shap"] = test_shap_dict
        results_dict["df_true_pred"] = test_df_true_pred_dict
        results_dict["attention"] = test_attention_dict

    results_dict["logging"] = {
        "metrics": logging_dict
    }

    if return_with_model:
        results_dict["model"] = model

    results_dict["model_structure"] = str(model)

    if config.args.callbacks.model_checkpoint.use_callback:
        results_dict["logging"]["checkpoints"] = {}
        for monitor_name, ckpt_path in check_ckpt_dict.items():
            monitor_prefix = f"_monitored_by_{monitor_name}" if monitor_name != monitor_key else ""
            results_dict["logging"]["checkpoints"][f"best{monitor_prefix}"] = ckpt_path["best"]
            results_dict["logging"]["checkpoints"][f"last{monitor_prefix}"] = ckpt_path["last"]

    print(f"Finished: {fold_val_num}")
    return results_dict
