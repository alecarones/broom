import broom.gilcs


def test_import_gilcs():
    assert hasattr(broom.gilcs, "__file__")


def test_gilc_runs():
    # Provide minimal dummy arguments
    broom.gilcs.gilc(None, None, None)
