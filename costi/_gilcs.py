import numpy as np
import healpy as hp
from .configurations import Configs
from .routines import _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering,\
                      _get_local_cov, _EB_to_QU, _E_to_QU, _B_to_QU, _save_compsep_products
from ._seds import _get_CMB_SED

def gilc(config: Configs, input_alms, compsep_run, nsim=None):
    compsep_run = _standardize_gnilc_run(compsep_run)

    output_maps = _gilc(config, input_alms, compsep_run)
    
    output_maps = _gilc_post_processing(config, output_maps, compsep_run)

    if config.save_compsep_products:
        _save_compsep_products(config, output_maps, compsep_run, nsim=nsim)
    if config.return_compsep_products:
        return output_maps

def _gilc(config: Configs, input_alms, compsep_run):
    if input_alms.ndim == 4:
        output_maps = np.zeros((input_alms.shape[1], 12 * config.nside**2, input_alms.shape[3]))
        for i in range(input_alms.shape[1]):
            output_maps[i] = _ilc_scalar(config, input_alms[:, i, :, :], compsep_run)
    elif input_alms.ndim == 3:
        output_maps = _ilc_scalar(config, input_alms, compsep_run)
    return output_maps

def _standardize_gnilc_run(config: Configs, compsep_run):
    if not "channels" in compsep_run:
        compsep_run["channels"] = np.arange(len(config.instrument.frequency))
        remove_items = True
    else: 
        remove_items = False
    return compsep_run, remove_items
