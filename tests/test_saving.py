from types import SimpleNamespace

import healpy as hp
import numpy as np

import broom.gilcs
import broom.saving
from broom.configurations import Configs
from broom.routines import merge_dicts


def test_import_saving():
    assert hasattr(broom.saving, "__file__")


def test_get_gnilc_maps_runs(config_all_path):
    config = Configs(config_all_path)
    compseps = config.compsep
    selected_compseps = [compsep for compsep in compseps if compsep["method"] == "gilc"]
    compsep = [selected_compseps[0]]

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
        broom.gilcs.gilc(config, input_alms, compsep)

    path_gnilc = "gilc_needlet_bias0.01/mexican_B1.3_j0j9_j10j13_j14j16_j17j33/"
    broom.saving.get_gnilc_maps(
        config=config, path_gnilc=path_gnilc, field_in=config.field_out, nsim="0"
    )
