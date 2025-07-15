from types import SimpleNamespace

import healpy as hp
import numpy as np

import broom.compsep
from broom.configurations import Configs


def test_import_compsep():
    assert hasattr(broom.compsep, "__file__")


def test_component_separation_runs(config_all_path):
    config = Configs(config_all_path)
    rng = np.random.default_rng()
    fgds = 1e-6 * np.ones(
        (len(config.instrument.frequency), 3, hp.Alm.getsize(config.lmax)),
        dtype=complex,
    )
    data = SimpleNamespace(
        total=1e-6
        * rng.random((len(config.instrument.frequency), 3, hp.Alm.getsize(config.lmax)))
        * (1 + 1j),
        noise=rng.random(
            (len(config.instrument.frequency), 3, hp.Alm.getsize(config.lmax))
        )
        * (1 + 1j),
        fgds=fgds,
        cmb=1e-6
        * np.ones(
            (len(config.instrument.frequency), 3, hp.Alm.getsize(config.lmax)),
            dtype=complex,
        ),
    )
    broom.clusters.get_and_save_real_tracers_B(config=config, foregrounds=fgds)
    broom.compsep.component_separation(config, data)


def test_get_data_and_compsep_runs(config_all_path):
    config = Configs(config_all_path)
    config.return_compsep_products = False
    foregrounds = SimpleNamespace(
        total=1e-6
        * np.ones(
            (len(config.instrument.frequency), 3, hp.Alm.getsize(config.lmax)),
            dtype=complex,
        )
    )
    broom.compsep.get_data_and_compsep(config, foregrounds)
