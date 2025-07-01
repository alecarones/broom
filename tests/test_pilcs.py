import pytest
import broom.pilcs

def test_import_pilcs():
    assert hasattr(broom.pilcs, "__file__")

if hasattr(broom.pilcs, "pilc"):
    def test_pilc_runs():
        # Provide minimal dummy arguments
        broom.pilcs.pilc(None, None, None)
