from types import SimpleNamespace

import healpy as hp
import numpy as np

import broom.templates
from broom.configurations import Configs


def test_import_templates():
    assert hasattr(broom.templates, "__file__")


def test_get_residuals_template_runs(config_simple_path):
    config = Configs(config_simple_path)
    # Minimal input_alms: (n_freqs, n_fields, n_alms, n_comps)
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
    compsep_run = {
        "gnilc_path": "gilc_needlet_bias0.01/mexican_B1.3_j0j9_j10j13_j14j16_j17j33",
        "compsep_path": "ilc_needlet_bias0.001/mexican_B1.3_j0j13_j14j16_j17j18_j19j39",
        "field_in": "B",
        "adapt_nside": False,
        "nsim": "0",
    }
    broom.templates.get_residuals_template(config, input_alms, compsep_run)


def test_get_fres_scalar_error(config_simple_path):
    config = Configs(config_simple_path)

    rng = np.random.default_rng()
    input_alms = SimpleNamespace(
        total=1e-6
        * rng.random((len(config.instrument.frequency), hp.Alm.getsize(config.lmax), 1))
        * (1 + 1j)
    )
    broom.templates._get_fres_scalar(config, input_alms.total, config.compsep[0])
