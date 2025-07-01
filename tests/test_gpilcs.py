import broom.gpilcs

def test_import_gpilcs():
    assert hasattr(broom.gpilcs, "__file__")

def test_gpilc_runs():
    # Provide minimal dummy arguments
    broom.gpilcs.gpilc(None, None, None)
