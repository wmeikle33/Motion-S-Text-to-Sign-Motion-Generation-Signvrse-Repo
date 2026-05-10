from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    random_state: int = 42
    artifacts_dir: Path = Path("models")
    model_path: Path = artifacts_dir / "models/model.keras"
    pipeline_path: Path = artifacts_dir / "pipeline.joblib"

