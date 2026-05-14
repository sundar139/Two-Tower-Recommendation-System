"""Typed configuration loading for the recommendation project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from movie_recsys.constants import PROJECT_ROOT


class OutputFilesConfig(BaseModel):
	"""Names for generated parquet and stats outputs."""

	interactions_train: str = "interactions_train.parquet"
	interactions_val: str = "interactions_val.parquet"
	interactions_test: str = "interactions_test.parquet"
	users: str = "users.parquet"
	items: str = "items.parquet"
	user_id_map: str = "user_id_map.parquet"
	item_id_map: str = "item_id_map.parquet"
	user_histories: str = "user_histories.parquet"
	dataset_stats: str = "dataset_stats.json"


class ProjectConfig(BaseModel):
	"""Global project configuration."""

	model_config = ConfigDict(extra="forbid")

	random_seed: int = 42


class DataConfig(BaseModel):
	"""Data-pipeline configuration with repository-relative directories."""

	model_config = ConfigDict(extra="forbid")

	raw_data_dir: Path = Path("data/raw")
	interim_data_dir: Path = Path("data/interim")
	processed_data_dir: Path = Path("data/processed")
	movielens_url: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
	expected_zip_name: str = "ml-25m.zip"
	expected_checksum: str | None = None
	positive_rating_threshold: float = 4.0
	min_positive_interactions_per_user: PositiveInt = 3
	sample_users: PositiveInt | None = None
	output_files: OutputFilesConfig = Field(default_factory=OutputFilesConfig)

	def resolve_paths(self, root: Path = PROJECT_ROOT) -> DataConfig:
		"""Return a copy of the config with absolute paths rooted at project root."""

		clone = self.model_copy(deep=True)
		clone.raw_data_dir = _resolve_path(clone.raw_data_dir, root)
		clone.interim_data_dir = _resolve_path(clone.interim_data_dir, root)
		clone.processed_data_dir = _resolve_path(clone.processed_data_dir, root)
		return clone


def _resolve_path(path_value: Path, root: Path) -> Path:
	if path_value.is_absolute():
		return path_value
	return (root / path_value).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as file_obj:
		data = yaml.safe_load(file_obj) or {}
	if not isinstance(data, dict):
		raise ValueError(f"Expected mapping in config file: {path}")
	return data


def load_project_config(config_path: str | Path = "configs/project.yaml") -> ProjectConfig:
	"""Load project config YAML into a typed model."""

	path = _resolve_path(Path(config_path), PROJECT_ROOT)
	return ProjectConfig.model_validate(_load_yaml(path))


def load_data_config(config_path: str | Path = "configs/data.yaml") -> DataConfig:
	"""Load data config YAML into a typed model and resolve paths."""

	path = _resolve_path(Path(config_path), PROJECT_ROOT)
	config = DataConfig.model_validate(_load_yaml(path))
	return config.resolve_paths(PROJECT_ROOT)
