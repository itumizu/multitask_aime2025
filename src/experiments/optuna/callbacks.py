import optuna
from omegaconf import DictConfig, ListConfig

class StopAfterCompletedTrialsCallback:
    def __init__(self, config: DictConfig | ListConfig):
        self.config = config
        self.max_n_trials = config.args.optuna.n_trials
        self.harf_trials = int(self.max_n_trials / 2)  # Halfway point

    def __call__(self, study, trial):
        trial_number = trial.number

        # Get the number of trials that are completed or currently running
        completed_trials = study.get_trials(
            states=[
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.RUNNING,
            ]
        )

        properties = {
            "Optuna status": {
                "multi_select": [{"name": f"Trial {trial_number}"}]
            }
        }

        # Stop optimization once the maximum number of trials is reached
        if len(completed_trials) >= self.max_n_trials:
            study.stop()
