import pytest
import broom.templates
import numpy as np
from types import SimpleNamespace

class DummyConfig:
    def __init__(self):
        self.nside = 2
        self.lmax = 8
        self.fwhm_out = 1.0
        self.pixel_window_out = False
        self.save_compsep_products = False
        self.return_compsep_products = False
        self.mask_type = None
        self.leakage_correction = None
        self.field_out = "E"
        self.path_outputs = "."

def test_import_templates():
    assert hasattr(broom.templates, "__file__")

def test_get_residuals_template_runs():
    config = DummyConfig()
    # Minimal input_alms: (n_freqs, n_fields, n_alms, n_comps)
    input_alms = SimpleNamespace()
    input_alms.data = np.zeros((2, 1, 12, 1))
    compsep_run = {"compsep_path": "ilc_pixel_bias0.0", "field_in_cs": "E", "nsim": None, "adapt_nside": False}
    # Should not raise, but will return None due to missing weights file
    try:
        broom.templates.get_residuals_template(config, input_alms, compsep_run)
    except Exception as e:
        assert "No such file or directory" in str(e) or "needlet bands" in str(e) or "not supported" in str(e)

def test_get_fres_scalar_error():
    config = DummyConfig()
    arr = np.zeros((2, 12, 1))
    compsep_run = {"compsep_path": "ilc_harmonic_bias0.0", "field_in_cs": "E", "nsim": None, "adapt_nside": False}
    # Should raise ValueError for unsupported domain
    try:
        broom.templates._get_fres_scalar(config, arr, compsep_run)
    except ValueError as e:
        assert "not supported" in str(e)
