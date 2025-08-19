from broom.configurations import Configs
import broom.inputs
import numpy as np
from types import SimpleNamespace


def test_import_inputs():
    assert hasattr(broom.inputs, "__file__")


def test_alms_from_data_runs(config_simple_path):
    config = Configs(config_simple_path)
    data = SimpleNamespace(total=1e-6 * np.ones((2, 3, 12)))
    # Should not raise
    broom.inputs._alms_from_data(config, data, "TQU")
