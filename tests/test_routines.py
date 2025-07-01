from types import SimpleNamespace

import healpy as hp
import numpy as np

import broom.routines
from broom.configurations import Configs


def test_import_routines():
    assert hasattr(broom.routines, "__file__")


def test_get_local_cov_runs(config_simple_path):
    config = Configs(config_simple_path)

    rng = np.random.default_rng()
    input_alms = SimpleNamespace(total=1e-6 * rng.random((3, 12 * config.nside**2)))

    broom.routines._get_local_cov(
        input_maps=input_alms.total,
        lmax=config.lmax,
        ilc_bias=config.compsep[0]["ilc_bias"],
    )
