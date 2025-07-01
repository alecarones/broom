from types import SimpleNamespace

import healpy as hp
import numpy as np

import broom.ilcs
from broom.configurations import Configs
from broom.routines import merge_dicts


def test_import_ilcs():
    assert hasattr(broom.ilcs, "__file__")


def test_ilc_runs(config_all_path):
    config = Configs(config_all_path)
    compseps = config.compsep
    selected_compseps = [compsep for compsep in compseps if compsep["method"] == "ilc"]

    rng = np.random.default_rng()
    input_alms = SimpleNamespace(
        total=1e-6
        * rng.random((len(config.instrument.frequency), 3, hp.Alm.getsize(config.lmax)))
        * (1 + 1j)
    )

    for compsep in selected_compseps:
        if compsep["domain"] == "needlet":
            compsep["needlet_config"] = merge_dicts(compsep["needlet_config"])
        config.compsep = [compsep]
        broom.ilcs.ilc(config, input_alms, compsep)

def test_cilc_runs(config_all_path):
    config = Configs(config_all_path)
    compseps = config.compsep
    selected_compseps = [compsep for compsep in compseps if compsep["method"] == "cilc" or compsep["method"] == "c_ilc"]

    rng = np.random.default_rng()
    input_alms = SimpleNamespace(
        total=1e-6
        * rng.random((len(config.instrument.frequency), 3, hp.Alm.getsize(config.lmax)))
        * (1 + 1j)
    )

    for compsep in selected_compseps:
        if compsep["domain"] == "needlet":
            compsep["needlet_config"] = merge_dicts(compsep["needlet_config"])
        compsep["constraints"] = merge_dicts(compsep["constraints"])
        config.compsep = [compsep]
        broom.ilcs.ilc(config, input_alms, compsep)
