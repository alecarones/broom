import broom.configurations


def test_import_configurations():
    assert hasattr(broom.configurations, "__file__")


def test_configs_instantiation(config_simple_path):
    cfg = broom.configurations.Configs(config_simple_path)
    assert cfg is not None
