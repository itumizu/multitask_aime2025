from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, ListConfig
import ulid


def setDefaltRootDir(config: DictConfig | ListConfig):
    default_root_dir = config.args.default_root_dir
    current_date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    default_root_dir = Path(default_root_dir, f"{current_date}_{str(ulid.ULID())}")

    return default_root_dir
