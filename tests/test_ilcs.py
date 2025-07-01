import pytest
import broom.ilcs

def test_import_ilcs():
    assert hasattr(broom.ilcs, "__file__")

if hasattr(broom.ilcs, "ilc"):
    def test_ilc_runs():
        # Provide minimal dummy arguments
        broom.ilcs.ilc(None, None, None)
