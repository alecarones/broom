import broom.needlets

def test_import_needlets():
    assert hasattr(broom.needlets, "__file__")

if hasattr(broom.needlets, "_get_needlet_windows_"):
    def test_get_needlet_windows_runs():
        # Provide minimal dummy arguments
        broom.needlets._get_needlet_windows_({}, 8)
