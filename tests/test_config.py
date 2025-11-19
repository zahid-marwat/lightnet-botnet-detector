from src.config import DEFAULT_CONFIG_PATH, ProjectConfig, load_config


def test_load_default_config():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(cfg, ProjectConfig)
    assert cfg.data.label_column == "attack_label"
    assert cfg.training.test_size > 0
