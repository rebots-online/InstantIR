from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SingleDataConfig:
    dataset_folder: str
    imagefolder: bool = True
    dataset_weight: float = 1.0 # Not used yet

@dataclass
class DataConfig:
    datasets: List[SingleDataConfig]
    val_dataset: Optional[SingleDataConfig] = None
