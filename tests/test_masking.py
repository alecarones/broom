import pytest
import broom.masking

def test_import_masking():
    assert hasattr(broom.masking, "__file__")

if hasattr(broom.masking, "get_masks_for_compsep"):
    def test_get_masks_for_compsep_runs():
        # Should not raise with None arguments
        broom.masking.get_masks_for_compsep(None, None, 2)
