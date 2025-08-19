from broom.configurations import Configs
import broom.needlets
from broom.routines import merge_dicts


def test_import_needlets():
    assert hasattr(broom.needlets, "__file__")


def test_get_needlet_windows_runs(config_all_path):
    config = Configs(config_all_path)
    compseps = config.compsep
    selected_compseps = [compsep for compsep in compseps if "needlet_config" in compsep]

    for compsep in selected_compseps:
        compsep["needlet_config"] = merge_dicts(compsep["needlet_config"])
        broom.needlets._get_needlet_windows_(compsep["needlet_config"], config.lmax)
