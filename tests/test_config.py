from __future__ import annotations

from pathlib import Path

from movie_recsys.config import load_data_config, load_project_config
from movie_recsys.constants import PROJECT_ROOT


def test_load_project_config_defaults() -> None:
	cfg = load_project_config()
	assert cfg.random_seed == 42


def test_load_data_config_resolves_paths() -> None:
	cfg = load_data_config()
	assert cfg.raw_data_dir == (PROJECT_ROOT / "data/raw").resolve()
	assert cfg.interim_data_dir == (PROJECT_ROOT / "data/interim").resolve()
	assert cfg.processed_data_dir == (PROJECT_ROOT / "data/processed").resolve()
	assert cfg.positive_rating_threshold == 4.0


def test_load_data_config_from_custom_file(tmp_path: Path) -> None:
	config_path = tmp_path / "data.yaml"
	config_path.write_text(
		"\n".join(
			[
				"raw_data_dir: data/raw",
				"interim_data_dir: data/interim",
				"processed_data_dir: data/processed",
				"movielens_url: https://files.grouplens.org/datasets/movielens/ml-25m.zip",
				"expected_zip_name: ml-25m.zip",
				"expected_checksum: null",
				"positive_rating_threshold: 4.5",
				"min_positive_interactions_per_user: 3",
			],
		),
		encoding="utf-8",
	)
	cfg = load_data_config(config_path)
	assert cfg.expected_zip_name == "ml-25m.zip"
	assert cfg.positive_rating_threshold == 4.5
