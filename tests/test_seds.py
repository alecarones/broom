import pytest
import broom.seds
import numpy as np

def test_import_seds():
    assert hasattr(broom.seds, "__file__")

if hasattr(broom.seds, "_get_CMB_SED"):
    def test_get_CMB_SED_runs():
        # Provide minimal dummy arguments
        broom.seds._get_CMB_SED(np.array([30.]), units="uK_CMB", bandwidths=None)
