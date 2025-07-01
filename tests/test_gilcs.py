import pytest
import broom.gilcs

def test_import_gilcs():
    assert hasattr(broom.gilcs, "__file__")

if hasattr(broom.gilcs, "gilc"):
    def test_gilc_runs():
        # Provide minimal dummy arguments
        broom.gilcs.gilc(None, None, None)
