import pytest
import broom.spectra
import numpy as np
from types import SimpleNamespace


def test_import_spectra():
    assert hasattr(broom.spectra, "__file__")


def test_standardize_compute_cls_error():
    class DummyConfig:
        field_out = "E"
        spectra_comp = "anafast"

    config = DummyConfig()
    # Missing path_method and components_for_cls
    with pytest.raises(ValueError):
        broom.spectra._standardize_compute_cls(config, {})
    # Missing components_for_cls
    with pytest.raises(ValueError):
        broom.spectra._standardize_compute_cls(config, {"path_method": "foo"})


def test_compute_spectra_typeerror():
    # _compute_spectra expects Configs instance
    with pytest.raises(TypeError):
        broom.spectra._compute_spectra(object())


# If a function is available, test it with dummy arguments
if hasattr(
    broom.spectra, "spectra_function_name"
):  # Replace with real function if known

    def test_spectra_function_runs():
        broom.spectra.spectra_function_name()
