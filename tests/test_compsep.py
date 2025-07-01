from types import SimpleNamespace

import numpy as np
import pytest

import broom.compsep
from broom.configurations import Configs


def test_import_compsep():
    assert hasattr(broom.compsep, "__file__")


def test_component_separation_runs(config_path):
    config = Configs(config_path)
    config.instrument.path_hits_maps = "broom/utils/norm_nHits_SA_35FOV_ns512.fits"
    config.mask_observations = "broom/utils/norm_nHits_SA_35FOV_ns512.fits"
    config.mask_covariance = None
    data = SimpleNamespace(total=1e-6 * np.ones((len(config.instrument.frequency), 3, 12*config.nside**2)))
    # Should not raise
    broom.compsep.component_separation(config, data)


def test_get_data_and_compsep_runs(config_path):
    config = Configs(config_path)
    config.return_compsep_products = False
    data = SimpleNamespace(total=1e-6 * np.ones((2, 3, 12)))
    # Should not raise
    broom.compsep.get_data_and_compsep(config, data)
