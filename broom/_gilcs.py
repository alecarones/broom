import numpy as np
import healpy as hp
from .configurations import Configs
from .routines import _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering,\
                      _get_local_cov, _EB_to_QU, _E_to_QU, _B_to_QU, _save_compsep_products,\
                      obj_to_array, array_to_obj, _get_full_path_out
from ._ilcs import get_ilc_cov, _get_good_channels_nl
from ._seds import _get_CMB_SED
import scipy
from numpy import linalg as lg
from types import SimpleNamespace
import os


def gilc(config: Configs, input_alms, compsep_run, **kwargs):
    compsep_run = _standardize_gnilc_run(compsep_run, input_alms.total.shape[0], config.lmax)

    if hasattr(input_alms, "nuisance"):
        nuis_alms = getattr(input_alms, "nuisance")
    else:
        if config.verbose:
            print("No nuisance alms provided. Using input CMB and noise alms as nuisance.")
        if compsep_run["cmb_nuisance"]:
            nuis_alms = getattr(input_alms, "cmb") + getattr(input_alms, "noise")
        else:
            nuis_alms = getattr(input_alms, "noise")

    output_maps = _gilc(config, obj_to_array(input_alms), nuis_alms, compsep_run, **kwargs)
    
    output_maps = _gilc_post_processing(config, output_maps, compsep_run, **kwargs)

    outputs = array_to_obj(output_maps, input_alms)
    del output_maps

    if config.save_compsep_products:
        _save_compsep_products(config, outputs, compsep_run, nsim=compsep_run["nsim"])
    
    if config.return_compsep_products:
        return outputs

def fgd_diagnostic(config: Configs, input_alms, compsep_run, **kwargs):
    if not "cmb_nuisance" in compsep_run:
        compsep_run["cmb_nuisance"] = True            
     
    if hasattr(input_alms, "nuisance"):
        nuis_alms = getattr(input_alms, "nuisance")
    else:
        if config.verbose:
            print("No nuisance alms provided. Using input CMB and noise alms as nuisance.")
        if compsep_run["cmb_nuisance"]:
            nuis_alms = getattr(input_alms, "cmb") + getattr(input_alms, "noise")
        else:
            nuis_alms = getattr(input_alms, "noise")

    output_maps = _fgd_diagnostic(config, input_alms.total, nuis_alms, compsep_run)

    outputs = SimpleNamespace()
    setattr(outputs, "m", output_maps)
    del output_maps

    if config.save_compsep_products:
        _save_compsep_products(config, outputs, compsep_run, nsim=compsep_run["nsim"])
    
    if config.return_compsep_products:
        return outputs

def _gilc_post_processing(config: Configs, output_maps, compsep_run, **kwargs):
    if (output_maps.ndim==4) and ((output_maps.shape[1]==2 and config.field_out == "QU") or (output_maps.shape[1]==3 and config.field_out == "TQU")):
        outputs = np.zeros_like(output_maps)
        for f, c in np.ndindex(output_maps.shape[0],output_maps.shape[-1]):
            outputs[f,...,c] = _EB_to_QU(output_maps[f,...,c],config.lmax, **kwargs)
        if hasattr(compsep_run, "mask"):
            outputs[:,:,compsep_run["mask"] == 0.,:] = 0.
        return outputs
    elif (output_maps.ndim==3) and (config.field_out in ["QU_E", "QU_B"]):
        output = np.zeros((output_maps.shape[0], 2, output_maps.shape[1], output_maps.shape[-1]))
        for f, c in np.ndindex(output_maps.shape[0],output_maps.shape[-1]):
            if config.field_out == "QU_E":
                output[f,...,c] = _E_to_QU(output_maps[f,:,c],config.lmax, **kwargs)
            elif config.field_out=="QU_B":
                output[f,...,c] = _B_to_QU(output_maps[f,:,c],config.lmax, **kwargs)
        if hasattr(compsep_run, "mask"):
            output[:,:,compsep_run["mask"] == 0.,:] = 0.
        return output
    else:
        if hasattr(compsep_run, "mask"):
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        return output_maps

def get_gnilc_maps(config: Configs, path_gnilc, field_in=None, nsim=None):
    if not os.path.exists(path_gnilc):
        raise ValueError(f"Path {path_gnilc} does not exist.")
    if field_in is None:
        field_in = config.field_out
    
    gnilc_maps = SimpleNamespace()

    if field_in in ["TQU", "TEB"]:
        if config.field_out == "T":
            gnilc_fields = 0
        elif config.field_out in ["QU", "QU_E", "QU_B", "E", "B"]:
            gnilc_fields = (1,2)
        elif config.field_out in ["TQU","TEB"]:
            gnilc_fields = (0,1,2)
    elif field_in in ["QU","EB"]:
        gnilc_fields = (0,1)
    elif field_in in ["T","E","B"]:
        gnilc_fields = 0

    if not os.path.exists(os.path.join(path_gnilc, "output_total")):
        raise ValueError(f"Path {path_gnilc} does not contain the expected multifrequency maps.")
    else:
        setattr(gnilc_maps, "total", [])
        for f, freq in enumerate(config.instrument.frequency):
            if nsim is None:
                filename = f"{path_gnilc}/output_total/{field_in}_output_total_{config.instrument.channels_tags[f]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}.fits"
            else:
                filename = f"{path_gnilc}/output_total/{field_in}_output_total_{config.instrument.channels_tags[f]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}_{nsim}.fits"
            getattr(gnilc_maps, "total").append(hp.read_map(filename, field=gnilc_fields))
        setattr(gnilc_maps, "total", np.array(getattr(gnilc_maps, "total")))

    if not os.path.exists(os.path.join(path_gnilc, "noise_residuals")):
        print(f"Warning: Path {path_gnilc} does not contain the expected noise residuals. Noise debias will not be possible")
    else:
        setattr(gnilc_maps, "noise", [])
        for f, freq in enumerate(config.instrument.frequency):
            if nsim is None:
                filename = f"{path_gnilc}/noise_residuals/{field_in}_noise_residuals_{config.instrument.channels_tags[f]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}.fits"
            else:
                filename = f"{path_gnilc}/noise_residuals/{field_in}_noise_residuals_{config.instrument.channels_tags[f]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}_{nsim}.fits"
            getattr(gnilc_maps, "noise").append(hp.read_map(filename, field=gnilc_fields))
        setattr(gnilc_maps, "noise", np.array(getattr(gnilc_maps, "noise")))

    if os.path.exists(os.path.join(path_gnilc, "fgds_residuals")):
        if config.verbose:
            print(f"Path {path_gnilc} does contain the expected foregrounds residuals. The ideal template of foregrounds residuals with no CMB and noise contamination will be computed")
        setattr(gnilc_maps, "fgds", [])
        for f, freq in enumerate(config.instrument.frequency):
            if nsim is None:
                filename = f"{path_gnilc}/fgds_residuals/{field_in}_fgds_residuals_{config.instrument.channels_tags[f]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}.fits"
            else:
                filename = f"{path_gnilc}/fgds_residuals/{field_in}_fgds_residuals_{config.instrument.channels_tags[f]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}_{nsim}.fits"
            getattr(gnilc_maps, "fgds").append(hp.read_map(filename, field=gnilc_fields))
        setattr(gnilc_maps, "fgds", np.array(getattr(gnilc_maps, "fgds")))

    return gnilc_maps

def _gilc(config: Configs, input_alms, nuis_alms, compsep_run, **kwargs):
    if input_alms.ndim == 4:
        output_maps = np.zeros((len(compsep_run["channels_out"]), input_alms.shape[1], 12 * config.nside**2, input_alms.shape[-1]))
        for i in range(input_alms.shape[1]):
            output_maps[:,i] = _gilc_scalar(config, input_alms[:, i], nuis_alms[:, i], compsep_run, **kwargs)
    elif input_alms.ndim == 3:
        output_maps = _gilc_scalar(config, input_alms, nuis_alms, compsep_run, **kwargs)
    return output_maps

def _fgd_diagnostic(config: Configs, input_alms, nuis_alms, compsep_run):
    if compsep_run["domain"]=="needlet":
        nls_number = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax).shape[0]

    if input_alms.ndim == 3:
        if compsep_run["domain"]=="needlet":
            output_maps = np.zeros((input_alms.shape[1], nls_number, 12 * config.nside**2))
        else:
            output_maps = np.zeros((input_alms.shape[1], 12 * config.nside**2))
        for i in range(input_alms.shape[1]):
            output_maps[i] = _fgd_diagnostic_scalar(config, input_alms[:, i], nuis_alms[:, i], compsep_run)

    elif input_alms.ndim == 2:
        output_maps = _fgd_diagnostic_scalar(config, input_alms, nuis_alms, compsep_run)

    return output_maps

def _standardize_gnilc_run(compsep_run, n_freqs, lmax):
    if not "channels_out" in compsep_run:
        compsep_run["channels_out"] = list(range(n_freqs))
    else: 
        if np.any(np.array(compsep_run["channels_out"]) >= n_freqs):
            raise ValueError("Some of the requested channels_out are not present in the input maps.")

    if compsep_run["domain"]=="needlet":
        nls_number = _get_needlet_windows_(compsep_run["needlet_config"], lmax).shape[0]
    
    if not "depro_cmb" in compsep_run:
        if compsep_run["domain"]=="pixel":
            compsep_run["depro_cmb"] = None
        elif compsep_run["domain"]=="needlet":
            compsep_run["depro_cmb"] = np.repeat(None, nls_number)
    else:
        if compsep_run["depro_cmb"] is not None:
            if not isinstance(compsep_run["depro_cmb"], (int, float, list, np.ndarray)):
                raise ValueError("depro_cmb must be a scalar or a list.")
            
        if (compsep_run["depro_cmb"] is None) and (compsep_run["domain"]=="needlet"):
            compsep_run["depro_cmb"] = np.repeat(None, nls_number)
        elif (isinstance(compsep_run["depro_cmb"], (int, float))) and (compsep_run["domain"]=="needlet"):
            compsep_run["depro_cmb"] = np.repeat(compsep_run["depro_cmb"], nls_number)
        elif (isinstance(compsep_run["depro_cmb"], (list, np.ndarray))) and (compsep_run["domain"]=="pixel"):
            raise ValueError("depro_cmb must be a scalar or None when domain is pixel.")
        elif (isinstance(compsep_run["depro_cmb"], list)) and (compsep_run["domain"]=="needlet"):
            if len(compsep_run["depro_cmb"]) < nls_number:
                while len(compsep_run["depro_cmb"]) < nls_number:
                    compsep_run["depro_cmb"].append(None)
            elif len(compsep_run["depro_cmb"]) > nls_number:
                compsep_run["depro_cmb"] = compsep_run["depro_cmb"][:nls_number]
        elif (isinstance(compsep_run["depro_cmb"], np.ndarray)) and (compsep_run["domain"]=="needlet"):
            if compsep_run["depro_cmb"].shape[0] < nls_number:
                while compsep_run["depro_cmb"].shape[0] < nls_number:
                    compsep_run["depro_cmb"] = np.append(compsep_run["depro_cmb"], None)
            elif compsep_run["depro_cmb"].shape[0] > nls_number:
                compsep_run["depro_cmb"] = compsep_run["depro_cmb"][:nls_number]

    if (not "m_bias" in compsep_run) or (compsep_run["m_bias"] is None):
        if compsep_run["domain"]=="pixel":
            compsep_run["m_bias"] = 0
        elif compsep_run["domain"]=="needlet":
            compsep_run["m_bias"] = np.repeat(0, nls_number)
    else:
        if not isinstance(compsep_run["m_bias"], (int, list, np.ndarray)):
            raise ValueError("m_bias must be a scalar, a list or a np.ndarray.")
        elif (isinstance(compsep_run["m_bias"], int)) and (compsep_run["domain"]=="needlet"):
            compsep_run["m_bias"] = np.repeat(compsep_run["m_bias"], nls_number)
        elif (isinstance(compsep_run["m_bias"], (list, np.ndarray))) and (compsep_run["domain"]=="pixel"):
            raise ValueError("m_bias must be a scalar when domain is pixel.")
        elif (isinstance(compsep_run["m_bias"], list)) and (compsep_run["domain"]=="needlet"):
            if len(compsep_run["m_bias"]) < nls_number:
                while len(compsep_run["m_bias"]) < nls_number:
                    compsep_run["m_bias"].append(0)
            elif len(compsep_run["m_bias"]) > nls_number:
                compsep_run["m_bias"] = compsep_run["m_bias"][:nls_number]    
        elif (isinstance(compsep_run["m_bias"], np.ndarray)) and (compsep_run["domain"]=="needlet"):
            if (compsep_run["m_bias"]).shape[0] < nls_number:
                while (compsep_run["m_bias"]).shape[0] < nls_number:
                    compsep_run["m_bias"] = np.append(compsep_run["m_bias"], 0)
            elif (compsep_run["m_bias"]).shape[0] > nls_number:
                compsep_run["m_bias"] = compsep_run["m_bias"][:nls_number]    

    if not "cmb_nuisance" in compsep_run:
        compsep_run["cmb_nuisance"] = True            
     
    return compsep_run

def _gilc_scalar(config: Configs, input_alms, nuis_alms, compsep_run, **kwargs):
    if compsep_run["domain"] == "pixel":
        output_maps = _gilc_pixel(config, input_alms, nuis_alms, compsep_run, **kwargs)
    elif compsep_run["domain"] == "needlet":
        output_maps = _gilc_needlet(config, input_alms, nuis_alms, compsep_run, **kwargs)
    return output_maps

def _fgd_diagnostic_scalar(config: Configs, input_alms, nuis_alms, compsep_run):
    if compsep_run["domain"] == "pixel":
        output_maps = _fgd_diagnostic_pixel(config, input_alms, nuis_alms, compsep_run)
    elif compsep_run["domain"] == "needlet":
        output_maps = _fgd_diagnostic_needlet(config, input_alms, nuis_alms, compsep_run)
    return output_maps

def _gilc_pixel(config: Configs, input_alms, nuis_alms, compsep_run, **kwargs):
    compsep_run["good_channels"] = _get_good_channels_nl(config, np.ones(lmax+1))
#    compsep_run["good_channels"] = np.arange(input_alms.shape[0])

    input_maps = np.zeros((compsep_run["good_channels"].shape[0], 12 * config.nside**2, input_alms.shape[-1]))
    nuis_maps = np.zeros((compsep_run["good_channels"].shape[0], 12 * config.nside**2))

    for n, channel in enumerate(compsep_run["good_channels"]):
#        if "mask" in compsep_run:
#            input_maps[n] = np.array([(hp.alm2map(np.ascontiguousarray(input_alms[channel, :, c]), config.nside, lmax=config.lmax, pol=False) * compsep_run["mask"]) for c in range(input_alms.shape[-1])]).T
#            nuis_maps[n] = (hp.alm2map(np.ascontiguousarray(nuis_alms[channel]), config.nside, lmax=config.lmax, pol=False) * compsep_run["mask"]) 
#        else:
        input_maps[n] = np.array([hp.alm2map(np.ascontiguousarray(input_alms[channel, :, c]), config.nside, lmax=config.lmax, pol=False) for c in range(input_alms.shape[-1])]).T
        nuis_maps[n] = hp.alm2map(np.ascontiguousarray(nuis_alms[channel]), config.nside, lmax=config.lmax, pol=False)

    output_maps = _gilc_maps(config, input_maps, nuis_maps, compsep_run, np.ones(config.lmax+1), depro_cmb=compsep_run["depro_cmb"], m_bias=compsep_run["m_bias"])
    del compsep_run['good_channels']

    if config.pixel_window_out:
        for f, c in np.ndindex(output_maps.shape[0],output_maps.shape[-1]):
            alm_out = hp.map2alm(output_maps[f,:,c],lmax=config.lmax, pol=False, **kwargs)
            output_maps[f,:,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=False, pixwin=True)
    if "mask" in compsep_run:
        output_maps[:,compsep_run["mask"] == 0.,:] = 0.
    return output_maps

def _fgd_diagnostic_pixel(config: Configs, input_alms, nuis_alms, compsep_run):
    input_maps = np.zeros((input_alms.shape[0], 12 * config.nside**2))
    nuis_maps = np.zeros((input_alms.shape[0], 12 * config.nside**2))

    for n in range(input_alms.shape[0]):
        input_maps[n] = hp.alm2map(np.ascontiguousarray(input_alms[n]), config.nside, lmax=config.lmax, pol=False)
        nuis_maps[n] = hp.alm2map(np.ascontiguousarray(nuis_alms[n]), config.nside, lmax=config.lmax, pol=False)

    output_maps = _get_diagnostic_maps(config, input_maps, nuis_maps, compsep_run, np.ones(config.lmax+1))

    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.] = 0.
    return output_maps

def _gilc_needlet(config: Configs, input_alms, nuis_alms, compsep_run, **kwargs):
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    if compsep_run["b_squared"]:
        b_ell = b_ell**2

    if compsep_run['needlet_config']['save_needlets']:
        path_out = _get_full_path_out(config, compsep_run)
        os.makedirs(path_out, exist_ok=True)
        np.save(os.path.join(path_out, "needlet_bands"), b_ell)
        
    output_alms = np.zeros((len(compsep_run["channels_out"]), input_alms.shape[1], input_alms.shape[-1]), dtype=complex)
    for j in range(b_ell.shape[0]):
        output_alms += _gilc_needlet_j(config, input_alms, nuis_alms, compsep_run, b_ell[j], depro_cmb=(compsep_run["depro_cmb"])[j], m_bias=(compsep_run["m_bias"])[j], **kwargs)
    
    output_maps = np.zeros((output_alms.shape[0], 12 * config.nside**2, output_alms.shape[-1]))
    for f, c in np.ndindex(output_alms.shape[0],output_alms.shape[-1]):
        output_maps[f,:,c] = hp.alm2map(np.ascontiguousarray(output_alms[f, :, c]), config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
    
    if "mask" in compsep_run:
        output_maps[:,compsep_run["mask"] == 0.,:] = 0.

    return output_maps

def _fgd_diagnostic_needlet(config: Configs, input_alms, nuis_alms, compsep_run):
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    if compsep_run["b_squared"]:
        b_ell = b_ell**2

    if compsep_run['needlet_config']['save_needlets']:
        path_out = _get_full_path_out(config, compsep_run)
        os.makedirs(path_out, exist_ok=True)
        np.save(os.path.join(path_out, "needlet_bands"), b_ell)
        
    output_maps = np.zeros((b_ell.shape[0], 12 * config.nside**2))
    for j in range(b_ell.shape[0]):
        output_maps[j] = _fgd_diagnostic_needlet_j(config, input_alms, nuis_alms, compsep_run, b_ell[j])
    
    if "mask" in compsep_run:
        output_maps[:,compsep_run["mask"] == 0.] = 0.

    return output_maps

def _gilc_needlet_j(config: Configs, input_alms, nuis_alms, compsep_run, b_ell, depro_cmb=None, m_bias=0, **kwargs):
    if "mask" in compsep_run:
        nside_, lmax_ = config.nside, config.lmax
    else:
        if compsep_run["adapt_nside"]:
            nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
        else:
            nside_, lmax_ = config.nside, config.lmax

    compsep_run["good_channels"] = _get_good_channels_nl(config, b_ell)
#    compsep_run["good_channels"] = np.arange(input_alms.shape[0])

    input_maps_nl = np.zeros((compsep_run["good_channels"].shape[0], 12 * nside_**2, input_alms.shape[-1]))
    nuis_maps_nl = np.zeros((compsep_run["good_channels"].shape[0], 12 * nside_**2))

    for n, channel in enumerate(compsep_run["good_channels"]):
        input_alms_j = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        nuis_alms_j = _needlet_filtering(nuis_alms[channel], b_ell, lmax_)
#        if "mask" in compsep_run:
#            input_maps_nl[n] = np.array([(hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False) * compsep_run["mask"]) for c in range(input_alms.shape[-1])]).T            
#            nuis_maps_nl[n] = hp.alm2map(np.ascontiguousarray(nuis_alms_j), nside_, lmax=lmax_, pol=False) * compsep_run["mask"]            
#        else:
        input_maps_nl[n] = np.array([hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False) for c in range(input_alms.shape[-1])]).T
        nuis_maps_nl[n] = hp.alm2map(np.ascontiguousarray(nuis_alms_j), nside_, lmax=lmax_, pol=False)

    output_maps_nl = _gilc_maps(config, input_maps_nl, nuis_maps_nl, compsep_run, b_ell, depro_cmb=depro_cmb, m_bias=m_bias)
    del input_maps_nl, nuis_maps_nl
    output_alms_j = np.zeros((output_maps_nl.shape[0], hp.Alm.getsize(config.lmax), output_maps_nl.shape[-1]), dtype=complex)
    for n in range(output_maps_nl.shape[0]):
        output_alms_nl = np.array([hp.map2alm(output_maps_nl[n,:,c], lmax=lmax_, pol=False, **kwargs) for c in range(output_maps_nl.shape[-1])]).T
        if compsep_run["b_squared"]:
            output_alms_j[n] = _needlet_filtering(output_alms_nl, np.ones(lmax_+1), config.lmax)
        else:
            output_alms_j[n] = _needlet_filtering(output_alms_nl, b_ell[:lmax_+1], config.lmax)

    del compsep_run['good_channels']

    return output_alms_j

def _fgd_diagnostic_needlet_j(config: Configs, input_alms, nuis_alms, compsep_run, b_ell):
    if "mask" in compsep_run:
        nside_, lmax_ = config.nside, config.lmax
    else:
        if compsep_run["adapt_nside"]:
            nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
        else:
            nside_, lmax_ = config.nside, config.lmax

#    compsep_run["good_channels"] = _get_good_channels_nl(config, b_ell)
    compsep_run["good_channels"] = np.arange(input_alms.shape[0])

    input_maps_nl = np.zeros((compsep_run["good_channels"].shape[0], 12 * nside_**2))
    nuis_maps_nl = np.zeros((compsep_run["good_channels"].shape[0], 12 * nside_**2))

    for n, channel in enumerate(compsep_run["good_channels"]):
        input_alms_j = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        nuis_alms_j = _needlet_filtering(nuis_alms[channel], b_ell, lmax_)
        input_maps_nl[n] = hp.alm2map(np.ascontiguousarray(input_alms_j), nside_, lmax=lmax_, pol=False)
        nuis_maps_nl[n] = hp.alm2map(np.ascontiguousarray(nuis_alms_j), nside_, lmax=lmax_, pol=False)

    output_maps_nl = _get_diagnostic_maps(config, input_maps_nl, nuis_maps_nl, compsep_run, b_ell)
    del input_maps_nl, nuis_maps_nl
    
    if hp.get_nside(output_maps_nl) < config.nside:
        output_maps_nl = hp.ud_grade(output_maps_nl, nside_out=config.nside)

    del compsep_run['good_channels']

    return output_maps_nl

def _gilc_maps(config: Configs, input_maps, nuis_maps, compsep_run, b_ell, depro_cmb=None, m_bias=0):

    cov = (get_ilc_cov(input_maps[...,0], config.lmax, compsep_run, b_ell)).T
    cov_n = (get_ilc_cov(nuis_maps, config.lmax, compsep_run, b_ell)).T

    λ, U = Cn_C_Cn(cov,cov_n)
    λ[λ<1.]=1.

    W = _get_gilc_weights(config, U, λ, cov, cov_n, input_maps.shape, compsep_run, depro_cmb=depro_cmb, m_bias=m_bias) 

    del cov, cov_n, U, λ

    if compsep_run["ilc_bias"] == 0.:
        output_maps = np.einsum('li,ijk->ljk', W, input_maps)
    else:
        output_maps = np.einsum('jli,ijk->ljk', W, input_maps)

    outputs = []
    for channel in compsep_run["channels_out"]:
        if channel in compsep_run["good_channels"]:
            outputs.append(output_maps[compsep_run["good_channels"] == channel][0])
        else:
            outputs.append(np.zeros(output_maps.shape[1:]))
    del output_maps

    return np.array(outputs)

def _get_diagnostic_maps(config: Configs, input_maps, nuis_maps, compsep_run, b_ell):

    cov = (get_ilc_cov(input_maps, config.lmax, compsep_run, b_ell)).T
    cov_n = (get_ilc_cov(nuis_maps, config.lmax, compsep_run, b_ell)).T

    λ, U = Cn_C_Cn(cov,cov_n)
    λ[λ<1.]=1.
    del cov, cov_n, U

    m = _get_gilc_m(λ) 

    if isinstance(m, (int, float)):
        m = np.repeat(m, input_maps.shape[-1])

    return m

def _get_gilc_weights(config: Configs, U, λ, cov, cov_n, input_shapes, compsep_run, depro_cmb=None, m_bias=0):

    if cov.ndim == 2:
        inv_cov = lg.inv(cov)

        m = _get_gilc_m(λ)

        if m_bias != 0:
            m = m + int(m_bias)  

        U_s = np.delete(U,np.where(λ < λ[m-1]),axis=1)
        λ_s = np.delete(λ,np.where(λ < λ[m-1]))

        F = lg.multi_dot([scipy.linalg.sqrtm(cov_n),U_s])
        
        if depro_cmb is None:
            W = lg.multi_dot([F,lg.inv(lg.multi_dot([F.T,inv_cov,F])),F.T,inv_cov])
        else:
            A_cmb = _get_CMB_SED(np.array(config.instrument.frequency), units=config.units)

            F_e = np.column_stack([F, depro_cmb * np.ones(input_shapes[0])])
            F_A = np.column_stack([F, A_cmb])
            W = lg.multi_dot([F_e,lg.inv(lg.multi_dot([F_A.T,inv_cov,F_A])),F_A.T,inv_cov])   

    elif cov.ndim == 3:
        m = _get_gilc_m(λ)
        
        if m_bias != 0:
            m = m + int(m_bias)  

        covn_sqrt = lg.cholesky(cov_n)
        covn_sqrt_inv = lg.inv(covn_sqrt)
        A_cmb = _get_CMB_SED(np.array(config.instrument.frequency), units=config.units)

        W_=np.zeros((cov.shape[0],input_shapes[0],input_shapes[0]))

        for m_ in np.unique(m):
            U_s=U[m==m_,:,:m_]
            cov_inv = lg.inv(cov[m==m_])
            F = np.einsum("kij,kjz->kiz", covn_sqrt[m==m_],U_s)

            if depro_cmb is None: 
                W_[m==m_] = np.einsum("kil,klj->kij",F,np.einsum("kzl,klj->kzj",lg.inv(np.einsum("kiz,kij,kjl->kzl",F,cov_inv,F)),np.einsum("kiz,kij->kzj",F,cov_inv)))
            else:
                e_cmb = depro_cmb * np.ones((F.shape[0],F.shape[1],1)) 
                F_e = np.concatenate((F,e_cmb),axis=2)
                F_A = np.concatenate((F,np.tile(A_cmb, (F.shape[0], 1))[:, :, np.newaxis]),axis=2)
                W_[m==m_] = np.einsum("kil,klj->kij",F_e,np.einsum("kzl,klj->kzj",lg.inv(np.einsum("kiz,kij,kjl->kzl",F_A,cov_inv,F_A)),np.einsum("kiz,kij->kzj",F_A,cov_inv)))

        if "mask" in compsep_run:
            W = np.zeros((input_shapes[1],W_.shape[1],W_.shape[2]))
            W[compsep_run["mask"] > 0.] = np.copy(W_)
        else:
            if W_.shape[0] != input_shapes[1]:
                W = np.zeros((input_shapes[1],W_.shape[1],W_.shape[2]))
                for i, k in np.ndindex(W_.shape[1],W_.shape[2]):
                    W[:,i,k]=hp.ud_grade(W_[:,i,k],hp.npix2nside(input_shapes[1]))
            else:
                W=np.copy(W_)

    return W

def _get_gilc_m(λ):
    if λ.ndim==1:
        A_m=np.zeros(λ.shape[0]+1)
        for i in range(λ.shape[0]):
            A_m[i]=2*i + np.sum((λ[i:]-np.log(λ[i:])-1.))
        A_m[λ.shape[0]]=2.*λ.shape[0]

        m = np.argmin(A_m)

    elif λ.ndim==2:
        A_m=np.zeros((λ.shape[0],λ.shape[1]+1))
        for i in range(λ.shape[1]):
            A_m[:,i]=2*i + np.sum((λ[:,i:]-np.log(λ[:,i:])-1.),axis=1)

        A_m[:,λ.shape[1]]=2.*λ.shape[1]
        m = np.argmin(A_m,axis=1)

    return m

def Cn_C_Cn(C,C_n):
    if (C.ndim == 2) and (C_n.ndim == 2):
        Cn_sqrt = lg.inv(scipy.linalg.sqrtm(C_n))
        M = lg.multi_dot([Cn_sqrt,C,Cn_sqrt])
    elif (C.ndim == 3) and (C_n.ndim == 3):
        Cn_sqrt = lg.inv(lg.cholesky(C_n))
        M = np.einsum("kij,kjz->kiz", Cn_sqrt, np.einsum("kij,kzj->kiz", C, Cn_sqrt))
    U, λ, _ = lg.svd(M)
    return λ, U
    