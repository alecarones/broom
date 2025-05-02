import numpy as np
import healpy as hp
from .configurations import Configs
from .routines import _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering,\
                      obj_to_array, array_to_obj, _save_residuals_template
from ._ilcs import _ilc_post_processing, _get_good_channels_nl
from types import SimpleNamespace
import os

def get_residuals_template(config: Configs, input_alms, compsep_run, nsim=None):
        
    templates = _get_fres(config, obj_to_array(input_alms), compsep_run)
    
    templates = _ilc_post_processing(config, templates, compsep_run)

    templates = array_to_obj(templates, input_alms)

    if config.save_compsep_products:
        _save_residuals_template(config, templates, compsep_run, nsim=nsim)
    if config.return_compsep_products:
        return templates

def _get_fres(config: Configs, input_alms, compsep_run):
    if input_alms.ndim == 4:
        if input_alms.shape[1] == 3:
            fields_ilc = ["T", "E", "B"]
        elif input_alms.shape[1] == 2:
            fields_ilc = ["E", "B"]
    elif input_alms.ndim == 3:
        if config.field_in in ["T", "E", "B"]:
            fields_ilc = [config.field_in]
        elif config.field_in in ["QU_E", "QU_B"]:
            fields_ilc = [config.field_in[-1]]

    if input_alms.ndim == 4:
        output_maps = np.zeros((input_alms.shape[1], 12 * config.nside**2, input_alms.shape[3]))
        for i in range(input_alms.shape[1]):
            compsep_run["field"] = fields_ilc[i]
            output_maps[i] = _get_fres_scalar(config, input_alms[:, i, :, :], compsep_run)
    elif input_alms.ndim == 3:
        compsep_run["field"] = fields_ilc[0]
        output_maps = _get_fres_scalar(config, input_alms, compsep_run)
    
    del compsep_run["field"]

    return output_maps

def _get_fres_scalar(config: Configs, input_alms, compsep_run):
    domain = compsep_run["compsep_path"].split('ilc_')[1].split('_bias')[0]
    if domain == "pixel":
        output_maps = _get_fres_pixel(config, input_alms, compsep_run)
    elif domain == "needlet":
        output_maps = _get_fres_needlet(config, input_alms, compsep_run)
    elif domain == "harmonic":
        output_maps = _get_fres_harmonic(config, input_alms, compsep_run)
    return output_maps

def _get_fres_needlet(config: Configs, input_alms, compsep_run):
    if os.path.exists(os.path.join(compsep_run["compsep_path"], "needlet_bands.npy")):
        b_ell = np.load(os.path.join(compsep_run["compsep_path"], "needlet_bands.npy"))
    else:
        raise ValueError(f"Needlet bands need to be saved as a npy file in {compsep_run['compsep_path']}")

    output_alms = np.zeros((input_alms.shape[1], input_alms.shape[-1]), dtype=complex)
    for j in range(b_ell.shape[0]):
        output_alms += _get_fres_needlet_j(config, input_alms, compsep_run, b_ell[j], j)
    
    output_maps = np.array([hp.alm2map(np.ascontiguousarray(output_alms[:, c]), config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out) for c in range(input_alms.shape[-1])]).T
    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.,:] = 0.
    return output_maps
        
def _get_fres_needlet_j(config: Configs, input_alms, compsep_run, b_ell, nl_scale):

    weights_filename = os.path.join(compsep_run["compsep_path"], "weights")
    weights_filename += f"/weights_{compsep_run['field']}_nl{nl_scale}.npy"

    w_ = np.load(weights_filename)

    if "mask" in compsep_run:
        nside_, lmax_ = config.nside, config.lmax
    else:
        if w_.ndim==1:
            nside_, lmax_ = config.nside, config.lmax
        elif w_.ndim==2:
            if hp.npix2nside(w_.shape[1]) < config.nside:
                nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
            else:
                nside_, lmax_ = config.nside, config.lmax            

    good_channels_nl = _get_good_channels_nl(config, b_ell)

    input_maps_nl = np.zeros((good_channels_nl.shape[0], 12 * nside_**2, input_alms.shape[-1]))
    for n, channel in enumerate(good_channels_nl):
        input_alms_j = _needlet_filtering(input_alms[channel], b_ell, lmax_)
    #    if "mask" in compsep_run:
    #        input_maps_nl[n] = np.array([(hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False) * compsep_run["mask"]) for c in range(input_alms.shape[-1])]).T            
    #    else:
        input_maps_nl[n] = np.array([hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False) for c in range(input_alms.shape[-1])]).T

    if w_.ndim==1:
        output_maps_nl = np.einsum('i,ijk->jk', w_, input_maps_nl)
    elif w_.ndim==2:
        output_maps_nl = np.einsum('ij,ijk->jk', w_, input_maps_nl)

    del input_maps_nl
    output_alms_nl = np.array([hp.map2alm(output_maps_nl[:,c], lmax=lmax_, pol=False) for c in range(output_maps_nl.shape[-1])]).T

    nl_bands = compsep_run["compsep_path"].rstrip('/').split('/')[-1]

    if "nlsquared" in nl_bands:
        output_alms_j = _needlet_filtering(output_alms_nl, np.ones(lmax_+1), config.lmax)
    else:
        output_alms_j = _needlet_filtering(output_alms_nl, b_ell[:lmax_+1], config.lmax)
    return output_alms_j

def _get_fres_pixel(config: Configs, input_alms, compsep_run):
    input_maps = np.zeros((input_alms.shape[0], 12 * config.nside**2, input_alms.shape[-1]))
    for n in range(input_alms.shape[0]):
#        if "mask" in compsep_run:
#            input_maps[n] = np.array([(hp.alm2map(np.ascontiguousarray(input_alms[n, :, c]), config.nside, lmax=config.lmax, pol=False) * compsep_run["mask"]) for c in range(input_alms.shape[-1])]).T
#        else:
        input_maps[n] = np.array([hp.alm2map(np.ascontiguousarray(input_alms[n, :, c]), config.nside, lmax=config.lmax, pol=False) for c in range(input_alms.shape[-1])]).T
    
    weights_filename = os.path.join(compsep_run["compsep_path"], "weights")
    weights_filename += f"/weights_{compsep_run['field']}.npy"

    w_ = np.load(weights_filename)
    
    if w_.ndim==1:
        output_maps = np.einsum('i,ijk->jk', w_, input_maps)
    elif w_.ndim==2:
        output_maps = np.einsum('ij,ijk->jk', w_, input_maps)

    if config.pixel_window_out:
        for c in range(output_maps.shape[1]):
            alm_out = hp.map2alm(output_maps[:,c],lmax=config.lmax, pol=False)
            output_maps[:,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=False, pixwin=True)
    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.,:] = 0.
    return output_maps

def _get_fres_maps(config: Configs, input_maps, compsep_run, b_ell, nl_scale=None):
    
    weights_filename = os.path.join(compsep_run["compsep_path"], "weights")
    weights_filename += f"/weights_{compsep_run['field']}"
    if nl_scale is not None:
        weights_filename += f"_nl{nl_scale}"
    weights_filename += ".npy"

    w_ = np.load(weights_filename)
    
    if w_.ndim==1:
        output_maps = np.einsum('i,ijk->jk', w_, input_maps)
    elif w_.ndim==2:
        output_maps = np.einsum('ij,ijk->jk', w_, input_maps)

    return output_maps
