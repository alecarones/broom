import broom.leakage
import numpy as np

def test_import_leakage():
    assert hasattr(broom.leakage, "__file__")

def test_leakage_function_runs():
    # Provide minimal dummy arguments
    broom.leakage.purify_master(np.zeros(10))
    broom.leakage.purify_recycling(np.zeros(10))
