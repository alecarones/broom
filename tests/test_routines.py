
import broom.routines
import numpy as np

def test_import_routines():
    assert hasattr(broom.routines, "__file__")

def test_get_local_cov_runs():
    # Provide minimal dummy arguments
    arr = np.ones((2, 10))
    broom.routines._get_local_cov(arr, 8, 0.0)
