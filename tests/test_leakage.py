import numpy as np

import broom.leakage
from broom.configurations import Configs


def test_import_leakage():
    assert hasattr(broom.leakage, "__file__")


def test_leakage_function_runs(config_simple_path):
    config = Configs(config_simple_path)
    QU_maps = 1e-6 * np.ones((2, 12 * config.nside**2))
    QU_full_maps = 1e-6 * np.ones((2, 12 * config.nside**2))
    mask = np.ones(12 * config.nside**2)

    broom.leakage.purify_master(QU_maps=QU_maps, mask=mask, lmax=config.lmax)
    broom.leakage.purify_recycling(
        QU_maps=QU_maps, QU_full_maps=QU_full_maps, mask=mask, lmax=config.lmax
    )
