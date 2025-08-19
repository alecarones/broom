import numpy as np

import broom.clusters
from broom.configurations import Configs


def test_import_clusters():
    assert hasattr(broom.clusters, "__file__")


def test_get_scalar_tracer_runs(config_simple_path):
    config = Configs(config_simple_path)
    broom.clusters.get_scalar_tracer(np.ones(12 * config.nside**2))
