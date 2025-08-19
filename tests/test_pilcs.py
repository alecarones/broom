from types import SimpleNamespace

import healpy as hp
import numpy as np

import broom.pilcs
from broom.configurations import Configs
from broom.routines import merge_dicts


def test_import_pilcs():
    assert hasattr(broom.pilcs, "__file__")


def test_pilc_runs(config_all_path):
    config = Configs(config_all_path)
    compseps = config.compsep
    selected_compseps = [
        compsep
        for compsep in compseps
        if compsep["method"] == "pilc" or compsep["method"] == "cpilc"
    ]

    rng = np.random.default_rng()
    input_alms = SimpleNamespace(
        total=1e-6
        * rng.random((len(config.instrument.frequency), 2, hp.Alm.getsize(config.lmax)))
        * (1 + 1j)
    )

    for compsep in selected_compseps:
        if compsep["domain"] == "needlet":
            compsep["needlet_config"] = merge_dicts(compsep["needlet_config"])
        if compsep["method"] == "cpilc":
            compsep["constraints"] = merge_dicts(compsep["constraints"])
        config.compsep = [compsep]
        broom.pilcs.pilc(config, input_alms, compsep)
