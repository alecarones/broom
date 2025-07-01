from broom.configurations import Configs
import broom.simulations


def test_import_simulations():
    assert hasattr(broom.simulations, "__file__")


def test_simulation_function_runs(config_simple_path):
    config = Configs(config_simple_path)
    broom.simulations._get_full_simulations(config)
