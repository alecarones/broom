import pytest
import broom.inputs
import numpy as np
from types import SimpleNamespace

def test_import_inputs():
    assert hasattr(broom.inputs, "__file__")

# If _alms_from_data is available, test it with dummy arguments
if hasattr(broom.inputs, "_alms_from_data"):
    def test_alms_from_data_runs():
        class DummyConfig:
            data_type = "maps"
            field_in = "TQU"
            field_out = "TQU"
            nside = 2
            lmax = 8
        config = DummyConfig()
        data = SimpleNamespace(total=1e-6 * np.ones((2, 3, 12)))
        # Should not raise
        broom.inputs._alms_from_data(config, data, "TQU")
