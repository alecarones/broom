import pytest
import broom.saving

def test_import_saving():
    assert hasattr(broom.saving, "__file__")

if hasattr(broom.saving, "get_gnilc_maps"):
    def test_get_gnilc_maps_runs():
        # Provide minimal dummy arguments
        broom.saving.get_gnilc_maps(None, None, None)
