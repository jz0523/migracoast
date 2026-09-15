from migracoast_ml import config


def test_states_are_the_five_covered_regions():
    assert config.STATES == ("MA", "NY", "PA", "FL", "CA")


def test_cell_geometry_matches_fema_rounding():
    assert config.CELL_SIZE == 0.1
    assert config.CELL_HALF == 0.05


def test_feature_names_has_eleven_entries_and_no_geography():
    assert len(config.FEATURE_NAMES) == 11
    assert len(set(config.FEATURE_NAMES)) == 11
    for banned in ("lat", "lon", "state"):
        assert not any(name == banned for name in config.FEATURE_NAMES)


def test_directories_are_under_the_repository_root():
    assert config.RAW_DIR.parts[-2:] == ("data", "raw")
    assert config.PROCESSED_DIR.parts[-2:] == ("data", "processed")
    assert config.MODELS_DIR.name == "models"


def test_ensure_dirs_creates_nested_directories_and_is_idempotent(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"

    monkeypatch.setattr(config, "RAW_DIR", raw_dir)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)

    assert not raw_dir.exists()
    assert not processed_dir.exists()
    assert not models_dir.exists()

    config.ensure_dirs()

    assert raw_dir.is_dir()
    assert processed_dir.is_dir()
    assert models_dir.is_dir()

    # Calling it again must not raise, even though the directories
    # (including their nested parents) already exist.
    config.ensure_dirs()

    assert raw_dir.is_dir()
    assert processed_dir.is_dir()
    assert models_dir.is_dir()
