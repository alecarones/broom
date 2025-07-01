from types import SimpleNamespace

import healpy as hp
import numpy as np

import broom.gpilcs
from broom.configurations import Configs
from broom.routines import merge_dicts


def test_import_gpilcs():
    assert hasattr(broom.gpilcs, "__file__")


def test_gpilc_runs(config_all_path):
    config = Configs(config_all_path)
    compseps = config.compsep
    selected_compseps = [
        compsep for compsep in compseps if compsep["method"] == "gpilc"
    ]

    rng = np.random.default_rng()
    input_alms = SimpleNamespace(
        total=1e-6
        * rng.random((len(config.instrument.frequency), 3, hp.Alm.getsize(config.lmax)))
        * (1 + 1j),
        noise=rng.random(
            (len(config.instrument.frequency), 3, hp.Alm.getsize(config.lmax))
        )
        * (1 + 1j),
        cmb=1e-6
        * np.ones(
            (len(config.instrument.frequency), 3, hp.Alm.getsize(config.lmax)),
            dtype=complex,
        ),
    )

    for compsep in selected_compseps:
        if compsep["domain"] == "needlet":
            compsep["needlet_config"] = merge_dicts(compsep["needlet_config"])
        config.compsep = [compsep]
        broom.gpilcs.gpilc(config, input_alms, compsep)
