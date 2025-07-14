import numpy as np
import healpy as hp
from .configurations import Configs
from .routines import _get_local_cov, _EB_to_QU, _E_to_QU, _B_to_QU, obj_to_array, array_to_obj, _log, _get_bandwidths
from .saving import _save_compsep_products, _get_full_path_out, save_ilc_weights
from .needlets import _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering, _get_good_channels_nl
from .pilcs import get_pilc_cov, get_prilc_cov
from .gilcs import _standardize_gnilc_run, Cn_C_Cn, _get_gilc_m, get_nuisance_idx
from .seds import _get_CMB_SED
import scipy
from numpy import linalg as lg
from types import SimpleNamespace
import os
from typing import Any, Dict, Optional, Union, List
import sys

def gpilc(config: Configs, input_alms: SimpleNamespace, compsep_run: Dict[str, Any], **kwargs) -> Optional[SimpleNamespace]:
    """
    Perform Generalized Polarization Internal Linear Combination (GPILC) on input spherical harmonics 
    to reconstruct Galactic emission at the observed microwave frequencies.

    Parameters
    ----------
        config : Configs
            Configuration object containing settings for the GPILC run. It should include:
                - lmax : int, maximum multipole for the component separation.
                - nside : int, HEALPix resolution parameter for compsep products.
                - fwhm_out : float, full width at half maximum of the output beam in arcminutes.
                - pixel_window_out : bool, whether to apply a pixel window to the output maps.
                - field_out : str, desired output fields (e.g., "E", "B", "QU", "EB", "QU_E", "QU_B").
                - save_compsep_products : bool, whether to save component separation products.
                - return_compsep_products : bool, whether to return component separation products.
                - path_outputs : str, path to save the output files.
                - leakage_correction : str, type of leakage correction to apply (e.g., "recycling", "purify").

        input_alms: SimpleNamespace
            Input multifrequency alms associated to polarization. 
            Each attribute should be a numpy array of shape (n_channels, 2, n_alms, n_components),
            if both E- and B-modes are provided, (n_channels, n_alms, n_components) otherwise.
            It should contain:
            - total : np.ndarray, alms of the total input maps.
            - nuisance : Optional[np.ndarray], alms of the nuisance maps (if available).
            Alternatively to nuisance, it can contain:
            - cmb : np.ndarray, alms of the CMB maps.
            - noise : np.ndarray, alms of the noise maps.

        compsep_run: Dict[str, Any]
            Dictionary with component separation parameters. It should include:
            - domain : str, either "pixel" or "needlet" for the component separation domain.
            - channels_out : list, indices of the frequency channels to reconstruct with GILC. Default is all channels.
            - depro_cmb : Optional[Union[float, list, np.ndarray]], deprojection factor for CMB (scalar or per needlet band). 
                    Default is None.
            - m_bias : Optional[Union[float, list, np.ndarray]], if not zero, it will include m_bias more (if m_bias > 0) 
                    or less (if m_bias < 0) modes in the reconstructed GPILC maps. Default is 0.
                    It can be a list if different values are needed for different needlet bands.
            - cmb_nuisance : bool, whether to include CMB alms in the nuisance covariance. Default is True.
            - needlet_config: Dictionary containing needlet settings. Needed if domain is "needlet". It should include:
                - "needlet_windows": Type of needlet windows ('cosine', 'standard', 'mexican').
                - "ell_peaks": List of integers defining multipoles of peaks for needlet bands (required for 'cosine').
                - "width": Width of the needlet windows (required for 'standard' and 'mexican').
                - "merging_needlets": Integer or list of integers defining ranges of needlets to be merged.
            - mask : Optional[np.ndarray], mask to apply to the maps (if available).
            - cov_noise_debias : Optional[Union[float, list, np.ndarray]], noise covariance debiasing factor.
                    If set to a non-zero value, it will subtract a 'noise_debias' fraction of noise covariance 
                    from the input and nuisance covariance matrices.
            
        **kwargs: 
            Dictionary of additional keyword arguments to pass to healpy function 'map2alm'.


    Returns
    -------
        Optional[SimpleNamespace] 
            Output object with component-separated maps, if config.return_compsep_products is True.
    """
    valid_fields = ["E", "B", "QU", "EB", "QU_E", "QU_B"]

    if config.field_out not in valid_fields:
        raise ValueError(f"Invalid field_out for GPILC. Must be one of: {', '.join(valid_fields)}.")

    compsep_run = _standardize_gnilc_run(compsep_run, input_alms.total.shape[0], config.lmax)

    compsep_run["nuis_idx"] = get_nuisance_idx(input_alms, compsep_run, config.verbose)
    if np.any(np.array(compsep_run["cov_noise_debias"]) != 0.):
        if not hasattr(input_alms, "noise"):
            raise ValueError("The input_alms object must have 'noise'' attribute for debiasing the covariance.")
        compsep_run["noise_idx"] = 2 if hasattr(input_alms, "fgds") else 1

    output_maps = _gpilc(config, obj_to_array(input_alms), compsep_run, **kwargs)
    
    outputs = array_to_obj(output_maps, input_alms)

    del output_maps
    compsep_run.pop("nuis_idx", None)
    compsep_run.pop("noise_idx", None)

    if config.save_compsep_products:
        _save_compsep_products(config, outputs, compsep_run, nsim=compsep_run["nsim"])
    
    if config.return_compsep_products:
        return outputs
    return None

def fgd_P_diagnostic(config: Configs, input_alms: SimpleNamespace, compsep_run: Dict[str, Any], **kwargs) -> Optional[SimpleNamespace]:
    """
    Perform foreground diagnostic for polarization (P) on input spherical harmonic coefficients (alms).

    Parameters
    ----------
        config: Configs
            Configuration object with general settings. It includes:
                - lmax : int, maximum multipole for the component separation.
                - nside : int, HEALPix resolution parameter for compsep products.
                - fwhm_out : float, full width at half maximum of the output beam in arcminutes.
                - field_out : str, desired fields to be considered (e.g., "T", "E", "B", "TEB").
                - save_compsep_products : bool, whether to save diagnostic maps.
                - return_compsep_products : bool, whether to return diagnostic maps.
                - path_outputs : str, path to save the output files.
        
        input_alms: SimpleNamespace
            Input multifrequency alms of polarization fields.
            Each attribute should be a numpy array of shape (n_channels, 2, n_alms, n_components),
            if both E- and B-modes are provided, (n_channels, n_alms, n_components) otherwise.
            It should contain:
            - total : np.ndarray, alms of the total input maps.
            - nuisance : Optional[np.ndarray], alms of the nuisance maps (if available).
            Alternatively to nuisance, it can contain:
            - cmb : np.ndarray, alms of the CMB maps.
            - noise : np.ndarray, alms of the noise maps.
    
        compsep_run: Dict[str, Any]
            Dictionary with diagnostic parameters. It should include:
                - domain : str, either "pixel" or "needlet" for the component separation domain.
                - cmb_nuisance : bool, whether to include CMB alms in the nuisance covariance. Default is True.
                - needlet_config: Dictionary containing needlet settings. Needed if domain is "needlet". It should include:
                    - "needlet_windows": Type of needlet windows ('cosine', 'standard', 'mexican').
                    - "ell_peaks": List of integers defining multipoles of peaks for needlet bands (required for 'cosine').
                    - "width": Width of the needlet windows (required for 'standard' and 'mexican').
                    - "merging_needlets": Integer or list of integers defining ranges of needlets to be merged.
                - adapt_nside : bool, whether to adapt the nside based on the needlet windows. Default is False.
                - mask : Optional[np.ndarray], mask to apply to the maps (if available).
                - cov_noise_debias : Optional[Union[float, list, np.ndarray]], noise covariance debiasing factor.
        
        **kwargs: 
            Dictionary of additional keyword arguments to pass to healpy function 'map2alm'.

    Returns
    -------
        Optional[SimpleNamespace]
            Object containing foreground diagnostic maps.
    """

    valid_fields = ["E", "B", "QU", "EB", "QU_E", "QU_B"]
    if config.field_out not in valid_fields:
        raise ValueError(f"Invalid field_in for foreground diagnostic. Must be one of: {', '.join(valid_fields)}")

    compsep_run.setdefault("cmb_nuisance", True)

    compsep_run["nuis_idx"] = get_nuisance_idx(input_alms, compsep_run, config.verbose)
    if np.any(np.array(compsep_run["cov_noise_debias"]) != 0.):
        if not hasattr(input_alms, "noise"):
            raise ValueError("The input_alms object must have 'noise'' attribute for debiasing the covariance.")
        compsep_run["noise_idx"] = 2 if hasattr(input_alms, "fgds") else 1

    if isinstance(compsep_run["nuis_idx"], int):
        nuis_alms = (obj_to_array(input_alms))[...,compsep_run["nuis_idx"]]
    elif isinstance(compsep_run["nuis_idx"], list):
        nuis_alms = (obj_to_array(input_alms))[...,compsep_run["nuis_idx"][0]] + (obj_to_array(input_alms))[...,compsep_run["nuis_idx"][1]]
    inputs_alms_for_diagn = np.concatenate([
        input_alms.total[...,np.newaxis],
        nuis_alms[...,np.newaxis]],axis=-1)
    del nuis_alms

    if np.any(np.array(compsep_run["cov_noise_debias"]) != 0.):
        noi_alms = (obj_to_array(input_alms))[...,compsep_run["noise_idx"]]
        inputs_alms_for_diagn = np.concatenate([inputs_alms_for_diagn, noi_alms[...,np.newaxis]], axis=-1)
        del noi_alms

    output_maps = _fgd_P_diagnostic(config, inputs_alms_for_diagn, compsep_run)
    del inputs_alms_for_diagn
    
    outputs = SimpleNamespace()
    setattr(outputs, "m", output_maps)
    del output_maps

    compsep_run.pop("nuis_idx", None)
    compsep_run.pop("noise_idx", None)

    if config.save_compsep_products:
        _save_compsep_products(config, outputs, compsep_run, nsim=compsep_run["nsim"])
    
    if config.return_compsep_products:
        return outputs
    return None

def _gpilc(config: Configs, input_alms: np.ndarray, compsep_run: Dict[str, Any], **kwargs) -> np.ndarray:
    """
    Perform Generalized Polarization Internal Linear Combination (GPILC) on input spherical harmonics.

    Parameters
    ----------
        config: Configs
            Configuration object with settings for the GPILC run.
        input_alms: np.ndarray
            Input multifrequency alms associated to polarization. 
            Shape should be (n_channels, 2, n_alms, n_components) if both E- and B-modes are provided, or (n_channels, n_alms, n_components) otherwise.
        compsep_run: Dict[str, Any]
            Dictionary with component separation parameters. See `gpilc` function for details.
        **kwargs:
            Additional keyword arguments to pass to healpy function 'map2alm'.

    Returns
    -------
        np.ndarray
            Output maps after performing GPILC. Shape will depend on the input alms and compsep_run settings.
    """
    if input_alms.ndim == 4:
        if input_alms.shape[1] != 2:
            raise ValueError("input_alms must have shape (nfreq, 2, nalm, ncomps) for gpilc.")
        compsep_run["field"] = "QU"

    elif input_alms.ndim == 3:
        if config.field_out in ["E", "QU_E"]:
            compsep_run["field"] = "QU_E"
        elif config.field_out in ["B", "QU_B"]:
            compsep_run["field"] = "QU_B"

    
    if compsep_run["domain"] == "pixel":
        return _gpilc_pixel(config, input_alms, compsep_run, **kwargs)
    elif compsep_run["domain"] == "needlet":
        return _gpilc_needlet(config, input_alms, compsep_run, **kwargs)
    else:
        raise ValueError(f"Invalid domain '{compsep_run['domain']}' for GPILC. Must be 'pixel' or 'needlet'.")

def _fgd_P_diagnostic(config: Configs, input_alms: np.ndarray, compsep_run: Dict[str, Any]) -> np.ndarray:
    """
    Perform foreground diagnostic for polarization (P) on input spherical harmonic coefficients (alms) based on domain configuration.

    Parameters
    ----------
        config: Configs
            Configuration object with settings for the diagnostic run. See `fgd_P_diagnostic` function for details.
        input_alms: np.ndarray
            Input multifrequency alms of polarization fields. 
            Shape should be (n_channels, 2, n_alms) if both E- and B-modes are provided, or (n_channels, n_alms) otherwise.
        compsep_run: Dict[str, Any]
            Dictionary with diagnostic parameters. See `fgd_P_diagnostic` function for details.
    
    Returns
    -------
        np.ndarray
            Foreground diagnostic maps. Shape will depend on the input alms and compsep_run settings.
    """

    if compsep_run["domain"] == "pixel":
        return _fgd_P_diagnostic_pixel(config, input_alms, compsep_run)
    elif compsep_run["domain"] == "needlet":
        return _fgd_P_diagnostic_needlet(config, input_alms, compsep_run)
    else:
        raise ValueError(f"Invalid domain '{compsep_run['domain']}' for foreground diagnostic. Must be 'pixel' or 'needlet'.")

def _gpilc_pixel(config: Configs, input_alms: np.ndarray, compsep_run: Dict[str, Any], **kwargs):
    """
    Perform Generalized Polarization Internal Linear Combination (GPILC) on input spherical harmonics in pixel space.

    Parameters
    ----------
        config: Configs
            Configuration object with settings for the GPILC run. See `gpilc` function for details.
        input_alms: np.ndarray
            Input multifrequency alms associated to polarization. 
            Shape should be (n_channels, 2, n_alms, n_components) if both E- and B-modes are provided, or (n_channels, n_alms, n_components) otherwise.
        compsep_run: Dict[str, Any]
            Dictionary with component separation parameters. See `gpilc` function for details.
        **kwargs:
            Additional keyword arguments to pass to healpy function 'map2alm'.

    Returns
    -------
        np.ndarray
            Output maps after performing GPILC in pixel space. Shape will depend on the input alms and compsep_run settings.
    """

    compsep_run["good_channels"] = _get_good_channels_nl(config, np.ones(config.lmax+1))
#    compsep_run["good_channels"] = np.arange(input_alms.shape[0])

    input_maps = np.zeros((compsep_run["good_channels"].shape[0], 2, 12 * config.nside**2, input_alms.shape[-1]))

    def alm_to_polmap(E=None, B=None):
        T = np.zeros_like(E if E is not None else B)
        return hp.alm2map([T, E if E is not None else T, B if B is not None else T],
                          config.nside, lmax=config.lmax, pol=True)[1:]

    for n, channel in enumerate(compsep_run["good_channels"]):
        for c in range(input_alms.shape[-1]):
            if input_alms.ndim == 4:
                input_maps[n, ..., c] = alm_to_polmap(E=input_alms[channel, 0, :, c],
                                                      B=input_alms[channel, 1, :, c])
            elif input_alms.ndim == 3:
                if config.field_out in ["QU_E", "E"]:
                    input_maps[n, ..., c] = alm_to_polmap(E=input_alms[channel, :, c])
                elif config.field_out in ["QU_B", "B"]:
                    input_maps[n, ..., c] = alm_to_polmap(B=input_alms[channel, :, c])

    # Perform GPILC separation
    output_maps = _gpilc_maps(
        config, input_maps, compsep_run,
        np.ones(config.lmax + 1), depro_cmb=compsep_run["depro_cmb"],
        m_bias=compsep_run["m_bias"], noise_debias=compsep_run["cov_noise_debias"],
    )

    if (config.field_out in ["QU", "QU_E", "QU_B"] and config.pixel_window_out) or (config.field_out in ["E","B"]) or (config.field_out=="EB"):
        for f, c in np.ndindex(output_maps.shape[0],output_maps.shape[-1]):
#            if ("mask" in compsep_run) and (config.mask_type == "observed_patch"):
#                if "_purify" in config.leakage_correction:
#                    alm_out = purify_master(output_maps[f,...,c], compsep_run["mask"], config.lmax,
#                                            purify_E=("E" in config.leakage_correction))
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
#                elif "_recycling" in config.leakage_correction:
#                    iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1)) \
#                        if "_iterations" in config.leakage_correction else 0
#                    alm_out = purify_recycling(output_maps[f, ..., c],
#                                               output_maps[f, ..., 0],
#                                               compsep_run["mask"],
#                                               config.lmax,
#                                               purify_E=("E" in config.leakage_correction),
#                                               iterations=iterations, **kwargs)
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
#               else:
#                   alm_out = hp.map2alm(np.array([0. * output_maps[f, 0, :, c],
#                                            output_maps[f, 0, :, c],
#                                            output_maps[f, 1, :, c]]) * compsep_run["mask"],lmax=config.lmax, pol=True, **kwargs)
#            else:
            alm_out = hp.map2alm([0. * output_maps[f, 0, :, c],
                                        output_maps[f, 0, :, c],
                                        output_maps[f, 1, :, c]],
                                        lmax=config.lmax, pol=True, **kwargs)

            if (config.field_out in ["QU", "QU_E", "QU_B"]):
                output_maps[f,...,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=True,
                                                    pixwin=config.pixel_window_out)[1:]
            elif config.field_out in ["E","B"]:
                output_maps[f,0,:,c] = hp.alm2map(alm_out[1] if config.field_out=="E" else alm_out[2],
                                                    config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
            elif config.field_out=="EB":
                output_maps[f,0,:,c] = hp.alm2map(alm_out[1], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
                output_maps[f,1,:,c] = hp.alm2map(alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)

        if config.field_out in ["E","B"]:
            output_maps = output_maps[:,0]

    del compsep_run['good_channels']
    if "mask" in compsep_run:
        if output_maps.ndim == 3:
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        elif output_maps.ndim == 4:
            output_maps[...,compsep_run["mask"] == 0.,:] = 0.

    return output_maps

def _fgd_P_diagnostic_pixel(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: Dict[str, Any]
) -> np.ndarray:
    """
    Perform foreground diagnostic for polarization (P) on input spherical harmonic coefficients (alms) in pixel space.

    Parameters
    ----------
        config: Configs
            Configuration object with settings for the diagnostic run. See `fgd_P_diagnostic` function for details.
        input_alms: np.ndarray
            Input multifrequency alms of polarization fields.
            Shape should be (n_channels, 2, n_alms) if both E- and B-modes are provided, or (n_channels, n_alms) otherwise.
        compsep_run: Dict[str, Any]
            Dictionary with diagnostic parameters. See `fgd_P_diagnostic` function for details.

    Returns
    -------
        np.ndarray
            Foreground diagnostic maps in pixel space. Shape will depend on the input alms and compsep_run settings.

    """
    compsep_run["good_channels"] = _get_good_channels_nl(config, np.ones(config.lmax+1))
    
    input_maps = np.zeros((compsep_run["good_channels"].shape[0], 2, 12 * config.nside**2, input_alms.shape[-1]))
    
    def alm_to_polmap(E=None, B=None):
        T = np.zeros_like(E if E is not None else B)
        return hp.alm2map(np.ascontiguousarray(np.array([T, E if E is not None else T, B if B is not None else T])),
                          config.nside, lmax=config.lmax, pol=True)[1:]

    for n, channel in enumerate(compsep_run["good_channels"]):
        for c in range(input_alms.shape[-1]):
            if input_alms.ndim == 4:
                input_maps[n, ..., c] = alm_to_polmap(E=input_alms[channel, 0, :, c],
                                                      B=input_alms[channel, 1, :, c])
            elif input_alms.ndim == 3:
                if config.field_out in ["QU_E", "E"]:
                    input_maps[n, ..., c] = alm_to_polmap(E=input_alms[channel, :, c])
                elif config.field_out in ["QU_B", "B"]:
                    input_maps[n, ..., c] = alm_to_polmap(B=input_alms[channel, :, c])

    output_maps = _fgd_P_diagnostic_maps(config, input_maps, compsep_run, np.ones(config.lmax+1), noise_debias=compsep_run["cov_noise_debias"])

    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.] = 0.
        
    return output_maps

def _gpilc_needlet(config: Configs,
                   input_alms: np.ndarray,
                   compsep_run: Dict[str, Any],
                   **kwargs) -> np.ndarray:
    """
    Perform Generalized Polarization Internal Linear Combination (GPILC) on input spherical harmonics in needlet space.

    Parameters
    ----------
        config: Configs
            Configuration object with settings for the GPILC run. See `gpilc` function for details.
        input_alms: np.ndarray
            Input multifrequency alms associated to polarization.
            Shape should be (n_channels, 2, n_alms, n_components) if both E- and B-modes are provided, or (n_channels, n_alms, n_components) otherwise.
        compsep_run: Dict[str, Any]
            Dictionary with component separation parameters. See `gpilc` function for details.
        **kwargs:
            Additional keyword arguments to pass to healpy function 'map2alm'.
    
    Returns
    -------
        np.ndarray
            Output maps after performing GPILC in needlet space. Shape will depend on the input alms and compsep_run settings.
    """

    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    b_ell = b_ell**2

    if compsep_run['save_needlets']:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        os.makedirs(compsep_run["path_out"], exist_ok=True)
        np.save(os.path.join(compsep_run["path_out"], "needlet_bands"), b_ell)

    output_maps = np.zeros((len(compsep_run["channels_out"]), 2, hp.nside2npix(config.nside), input_alms.shape[-1]))
    
    for j, b_ell_j in enumerate(b_ell):
        output_maps += _gpilc_needlet_j(config, input_alms, compsep_run,
                                        b_ell_j, j,
                                        depro_cmb=compsep_run["depro_cmb"][j],
                                        m_bias=compsep_run["m_bias"][j],
                                        noise_debias=compsep_run["cov_noise_debias"][j]
                                        )

    
    if ((config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out) or (config.field_out in ["E","B"]) or (config.field_out=="EB"):
        for f, c in np.ndindex(output_maps.shape[0],output_maps.shape[-1]):
#            if ("mask" in compsep_run) and (config.mask_type == "observed_patch") and config.leakage_correction is not None:
#                if "_purify" in config.leakage_correction:
#                    alm_out = purify_master(output_maps[f, ..., c],
#                                            compsep_run["mask"],
#                                            config.lmax,
#                                            purify_E=("E" in config.leakage_correction))
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
#                elif "_recycling" in config.leakage_correction:
#                    iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1)) \
#                        if "_iterations" in config.leakage_correction else 0
#                    alm_out = purify_recycling(output_maps[f, ..., c],
#                                            output_maps[f, ..., 0],
#                                            compsep_run["mask"],
#                                            config.lmax,
#                                            purify_E=("E" in config.leakage_correction),
#                                            iterations=iterations, **kwargs)
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
#                elif config.leakage_correction=="mask_only":
#                    alm_out = hp.map2alm(np.array([0. * output_maps[f, 0, :, c],
#                                            output_maps[f, 0, :, c],
#                                            output_maps[f, 1, :, c]]) * compsep_run["mask"],lmax=config.lmax, pol=True, **kwargs)
#            else:
            alm_out = hp.map2alm([0. * output_maps[f, 0, :, c],
                                        output_maps[f, 0, :, c],
                                        output_maps[f, 1, :, c]],
                                        lmax=config.lmax, pol=True, **kwargs)

            if (config.field_out in ["QU", "QU_E", "QU_B"]):
                output_maps[f, ..., c] = hp.alm2map(alm_out,
                                                    config.nside,
                                                    lmax=config.lmax,
                                                    pol=True,
                                                    pixwin=config.pixel_window_out)[1:]
            elif config.field_out in ["E","B"]:
                output_maps[f,0,:,c] = hp.alm2map(alm_out[1] if config.field_out=="E" else alm_out[2], config.nside, lmax=config.lmax,
                                                     pol=False, pixwin=config.pixel_window_out)
            elif config.field_out=="EB":
                output_maps[f, 0, :, c] = hp.alm2map(alm_out[1], config.nside, lmax=config.lmax,
                                                     pol=False, pixwin=config.pixel_window_out)
                output_maps[f, 1, :, c] = hp.alm2map(alm_out[2], config.nside, lmax=config.lmax,
                                                     pol=False, pixwin=config.pixel_window_out)

        if config.field_out in ["E","B"]:
            output_maps = output_maps[:,0]

    if "mask" in compsep_run:
        if output_maps.ndim == 3:
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        elif output_maps.ndim == 4:
            output_maps[...,compsep_run["mask"] == 0.,:] = 0.
    
    return output_maps
        
def _gpilc_needlet_j(config: Configs,
                     input_alms: np.ndarray,
                     compsep_run: Dict[str, Any],
                     b_ell: np.ndarray,
                     nl_scale: int,
                     depro_cmb: Optional[float] = None,
                     m_bias: Optional[Union[int, float]] = 0,
                     noise_debias: Optional[float] = 0.,
                     ) -> np.ndarray:
    """
    Perform GPILC on a single needlet band.

    Parameters
    ----------
        config: Configs
            Configuration object with settings for the GPILC run. See `gpilc` function for details.
        input_alms: np.ndarray
            Input multifrequency alms associated to polarization.
            Shape should be (n_channels, 2, n_alms, n_components) if both E- and B-modes are provided, or (n_channels, n_alms, n_components) otherwise.
        compsep_run: Dict[str, Any]
            Dictionary with component separation parameters. See `gpilc` function for details.
        b_ell: np.ndarray
            Needlet bandpass filter. Shape should be (lmax+1).
        nl_scale : int
            Needlet scale index corresponding to the current GPILC run. Used for saving weights with proper label.
        depro_cmb: float, optional
            Deprojection factor for CMB. If None, no deprojection is applied. 
            Otherwise CMB residuals in GPILC maps will be at the level of depro_cmb * CMB_input in the considered needlet band.
        m_bias: int or float, optional
            It will include m_bias more (if m_bias > 0) or less (if m_bias < 0) modes in the reconstructed GNILC maps.
        noise_debias: float, optional
            Noise debiasing factor. If set to a non-zero value, it will subtract a 'noise_debias' fraction of
            noise covariance from the input and nuisance covariance matrices.
    
    Returns
    -------
        np.ndarray
            Output maps after performing GPILC on the specified needlet band. Shape will depend on the input alms and compsep_run settings.
    """

    nside_, lmax_ = config.nside, config.lmax

    compsep_run["good_channels"] = _get_good_channels_nl(config, b_ell)
#    compsep_run["good_channels"] = np.arange(input_alms.shape[0])

    input_maps_nl = np.zeros((compsep_run["good_channels"].shape[0], 2, 12 * nside_**2, input_alms.shape[-1]))

    for n, channel in enumerate(compsep_run["good_channels"]):
        input_alms_j = np.zeros((2, hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)
        if input_alms.ndim == 4:
            for k in range(2):
                input_alms_j[k] = _needlet_filtering(input_alms[channel,k], b_ell, lmax_)
        elif input_alms.ndim == 3:
            if config.field_out in ["QU_E", "E"]:
                input_alms_j[0] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
            elif config.field_out in ["QU_B", "B"]:
                input_alms_j[1] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        
#        if ("mask" in compsep_run) and (config.mask_type == "observed_patch"):
#            for c in range(input_alms.shape[-1]):
#                input_maps_nl[n,...,c] = (hp.alm2map(np.ascontiguousarray([0. * input_alms_j[0, :, c],
#                                                    input_alms_j[0, :, c],
#                                                    input_alms_j[1, :, c]]),
#                                                   nside_, lmax=lmax_, pol=True)[1:]) * compsep_run["mask"]       
#        else:
        for c in range(input_alms.shape[-1]):
            input_maps_nl[n, ..., c] = hp.alm2map(np.ascontiguousarray([0. * input_alms_j[0, :, c],
                                                input_alms_j[0, :, c],
                                                input_alms_j[1, :, c]]),
                                                nside_, lmax=lmax_, pol=True)[1:]

    output_maps_nl = _gpilc_maps(config, input_maps_nl, compsep_run,
                              b_ell, depro_cmb=depro_cmb, m_bias=m_bias, noise_debias=noise_debias, nl_scale=nl_scale)

    del input_maps_nl
    del compsep_run['good_channels']

    return output_maps_nl

def _fgd_P_diagnostic_needlet(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: dict
) -> np.ndarray:
    """
    Perform diagnostic of foreground complexity in polarization intensity P and in needlet space.

    Parameters
    ----------
        config: Configs
            Configuration object with settings for the diagnostic run. See `fgd_P_diagnostic` function for details.
        input_alms: np.ndarray
            Input multifrequency alms of polarization fields.
            Shape should be (n_channels, 2, n_alms) if both E- and B-modes are provided, or (n_channels, n_alms) otherwise.
        compsep_run: Dict[str, Any]
            Dictionary with diagnostic parameters. See `fgd_P_diagnostic` function for details.
    
    Returns
    -------
        np.ndarray
            Foreground diagnostic maps in needlet space. Shape will depend on the input alms and compsep_run settings.
    """

    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    b_ell = b_ell**2

    if compsep_run['save_needlets']:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        os.makedirs(compsep_run["path_out"], exist_ok=True)
        np.save(os.path.join(compsep_run["path_out"], "needlet_bands"), b_ell)

    output_maps = np.zeros((b_ell.shape[0], hp.nside2npix(config.nside)))
    
    for j in range(b_ell.shape[0]):
        output_maps[j] = _fgd_P_diagnostic_needlet_j(config, input_alms, compsep_run, b_ell[j], 
                                                      noise_debias=compsep_run["cov_noise_debias"][j])
    
    if "mask" in compsep_run:
        output_maps[:,compsep_run["mask"] == 0.] = 0.
        
    return output_maps
        
def _fgd_P_diagnostic_needlet_j(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: dict,
    b_ell: np.ndarray,
    noise_debias: Optional[float] = 0.0
) -> np.ndarray:
    """
    Perform diagnostic of foreground complexity in polarization intensity P for a single needlet band.

    Parameters
    ----------
        config: Configs
            Configuration object with settings for the diagnostic run. See `fgd_P_diagnostic` function for details.
        input_alms: np.ndarray
            Input multifrequency alms of polarization fields.
            Shape should be (n_channels, 2, n_alms) if both E- and B-modes are provided, or (n_channels, n_alms) otherwise.
        compsep_run: Dict[str, Any]
            Dictionary with diagnostic parameters. See `fgd_P_diagnostic` function for details.
        b_ell: np.ndarray
            Needlet bandpass filter for the current band. Shape should be (lmax+1).
        noise_debias: float, optional
            Noise debiasing factor. If set to a non-zero value, it will subtract a 'noise_debias' fraction of
            noise covariance from the input and nuisance covariance matrices.
    
    Returns
    -------
        np.ndarray
            Foreground diagnostic maps in needlet space for the specified band. Shape will depend on the input alms and compsep_run settings.
    """

    if "mask" in compsep_run or not compsep_run["adapt_nside"]:
        nside_, lmax_ = config.nside, config.lmax
    else:
        nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
        
    compsep_run["good_channels"] = _get_good_channels_nl(config, b_ell)
#    compsep_run["good_channels"] = np.arange(input_alms.shape[0])

    input_maps_nl = np.zeros((compsep_run["good_channels"].shape[0], 2, 12 * nside_**2, input_alms.shape[-1]))

    for n, channel in enumerate(compsep_run["good_channels"]):
        input_alms_j = np.zeros((2, hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)
        if input_alms.ndim == 4:
            for k in range(2):
                input_alms_j[k] = _needlet_filtering(input_alms[channel,k], b_ell, lmax_)
        elif input_alms.ndim == 3:
            if config.field_out in ["QU_E", "E"]:
                input_alms_j[0] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
            elif config.field_out in ["QU_B", "B"]:
                input_alms_j[1] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        
        for c in range(input_alms.shape[-1]):
            input_maps_nl[n, ..., c] = hp.alm2map(np.ascontiguousarray([0. * input_alms_j[0, :, c],
                                                input_alms_j[0, :, c],
                                                input_alms_j[1, :, c]]),
                                                nside_, lmax=lmax_, pol=True)[1:]

    output_maps_nl = _fgd_P_diagnostic_maps(config, input_maps_nl, compsep_run, b_ell, noise_debias=noise_debias)
    del input_maps_nl
    del compsep_run['good_channels']

    if hp.get_nside(output_maps_nl) < config.nside:
        output_maps_nl = hp.ud_grade(output_maps_nl, config.nside)

    return output_maps_nl

def _gpilc_maps(
    config: Configs,
    input_maps: np.ndarray,
    compsep_run: dict,
    b_ell: np.ndarray,
    depro_cmb: Optional[float] = None,
    m_bias: Optional[Union[int, float]] = 0,
    noise_debias: Optional[float] = 0.0,
    nl_scale: Optional[Union[int, None]] = None,
) -> np.ndarray:
    """
    Perform Generalized Polarization Internal Linear Combination (GPILC) on input maps.

    Parameters
    ----------
        config: Configs
            Configuration object with settings for the GPILC run. See `gpilc` function for details.
        input_maps: np.ndarray
            Input multifrequency maps associated to polarization.
            Shape should be (n_channels, 2, n_pixels, n_components).
        compsep_run: dict
            Dictionary with component separation parameters. See `gpilc` function for details.
        b_ell: np.ndarray
            Bandpass filter for the GPILC. Shape should be (lmax+1).
            If compsep_run["domain"] is "pixel", it should be an array of ones.
        depro_cmb: float, optional
            Deprojection factor for CMB. If None, no deprojection is applied.
            Otherwise CMB residuals in GPILC maps will be at the level of depro_cmb * CMB_input.
        m_bias: int or float, optional
            It will include m_bias more (if m_bias > 0) or less (if m_bias < 0) modes in the reconstructed GPILC maps.
        noise_debias: float, optional
            Noise debiasing factor. If set to a non-zero value, it will subtract a 'noise_debias' fraction of
            noise covariance from the input and nuisance covariance matrices.
        nl_scale : int, optional
            Needlet scale index corresponding to the current GPILC run. Used for saving weights with proper label.

    Returns
    -------
        np.ndarray
            Output maps after performing GPILC. Shape is (n_channels_out, 2, n_pixels, n_components),
            where n_channels_out is the number of channels specified in compsep_run["channels_out"].
    """

    cov = (get_prilc_cov(input_maps[...,0], config.lmax, compsep_run, b_ell)).T

    if isinstance(compsep_run["nuis_idx"], int):
        cov_n = (get_prilc_cov(input_maps[...,compsep_run["nuis_idx"]], config.lmax, compsep_run, b_ell)).T
    elif isinstance(compsep_run["nuis_idx"], list):
        cov_n = (get_prilc_cov(input_maps[...,compsep_run["nuis_idx"][0]] + input_maps[...,compsep_run["nuis_idx"][1]], config.lmax, compsep_run, b_ell)).T

    if noise_debias != 0.:
        cov_noi = (get_prilc_cov(input_maps[...,compsep_run["noise_idx"]], config.lmax, compsep_run, b_ell)).T
        cov = cov - noise_debias * cov_noi
        cov_n = cov_n - noise_debias * cov_noi
        del cov_noi

    λ, U = Cn_C_Cn(cov,cov_n)
    λ[λ<1.]=1.

    W = _get_gpilc_weights(
        config, U, λ, cov, cov_n, input_maps.shape, compsep_run,
        depro_cmb=depro_cmb, m_bias=m_bias
    )
    if compsep_run["save_weights"]:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        save_ilc_weights(config, W, compsep_run,
                         hp.npix2nside(input_maps.shape[-2]), nl_scale=nl_scale)

    del cov, cov_n, U, λ

    if compsep_run["ilc_bias"] == 0.:
        if W.ndim==2:
            output_maps = np.einsum('li,ifjk->lfjk', W, input_maps)
        elif W.ndim==3:
            output_maps = np.zeros((W.shape[0], 2, *input_maps.shape[-2:]))
            output_maps[:, 0] = (
                np.einsum('li,ijk->ljk', W[0], input_maps[:, 0]) -
                np.einsum('li,ijk->ljk', W[1], input_maps[:, 1])
            )
            output_maps[:, 1] = (
                np.einsum('li,ijk->ljk', W[1], input_maps[:, 0]) +
                np.einsum('li,ijk->ljk', W[0], input_maps[:, 1])
            )
    else:
        if W.ndim==3:
            output_maps = np.einsum('jli,ifjk->lfjk', W, input_maps)
        elif W.ndim==4:
            output_maps = np.zeros((W.shape[0], 2, *input_maps.shape[-2:]))
            output_maps[:, 0] = (
                np.einsum('jli,ijk->ljk', W[0], input_maps[:, 0]) -
                np.einsum('jli,ijk->ljk', W[1], input_maps[:, 1])
            )
            output_maps[:, 1] = (
                np.einsum('jli,ijk->ljk', W[1], input_maps[:, 0]) +
                np.einsum('jli,ijk->ljk', W[0], input_maps[:, 1])
            )

    outputs = []
    for channel in compsep_run["channels_out"]:
        if channel in compsep_run["good_channels"]:
            outputs.append(output_maps[compsep_run["good_channels"] == channel][0])
        else:
            outputs.append(np.zeros(output_maps.shape[1:]))
    del output_maps

    return np.array(outputs)

def _fgd_P_diagnostic_maps(
    config: Configs,
    input_maps: np.ndarray,
    compsep_run: dict,
    b_ell: np.ndarray,
    noise_debias: Optional[float] = 0.0
) -> np.ndarray:
    """
    Perform diagnostic of foreground complexity in polarization intensity P.

    Parameters
    ----------
        config: Configs
            Configuration object with settings for the diagnostic run. See `fgd_P_diagnostic` function for details.
        input_maps: np.ndarray
            Input multifrequency maps associated to polarization.
            Shape should be (n_channels, 2, n_pixels).
        compsep_run: dict
            Dictionary with component separation parameters. See `fgd_P_diagnostic` function for details.
        b_ell: np.ndarray
            Needlet bandpass filter for the diagnostic. Shape should be (lmax+1).
            If compsep_run["domain"] is "pixel", it should be an array of ones.
        noise_debias: float, optional
            Noise debiasing factor. If set to a non-zero value, it will subtract a 'noise_debias' fraction of
            noise covariance from the input and nuisance covariance matrices.
    
    Returns
    -------
        np.ndarray
            Maps of foreground complexity in polarization intensity P.
            Shape is (n_pixels).
    """

    cov = (get_prilc_cov(input_maps[...,0], config.lmax, compsep_run, b_ell)).T
    cov_n = (get_prilc_cov(input_maps[...,1], config.lmax, compsep_run, b_ell)).T
    
    if noise_debias != 0.:
        cov_noi = (get_prilc_cov(input_maps[...,2], config.lmax, compsep_run, b_ell)).T
        cov = cov - noise_debias * cov_noi
        cov_n = cov_n - noise_debias * cov_noi
        del cov_noi

    λ, U = Cn_C_Cn(cov,cov_n)
    λ[λ<1.]=1.

    m = _get_gilc_m(λ)

    if isinstance(m, (int, float)):
        m = np.repeat(m, input_maps.shape[-2])

    return m

def _get_gpilc_weights(
    config: Configs,
    U: np.ndarray,
    λ: np.ndarray,
    cov: np.ndarray,
    cov_n: np.ndarray,
    input_shapes: tuple,
    compsep_run: dict,
    depro_cmb: Optional[float] = None,
    m_bias: Optional[Union[int, float]] = 0
) -> np.ndarray:
    """
    Compute the weights for Generalized Polarization Internal Linear Combination (GPILC).

    Parameters
    ----------
        config: Configs
            Configuration object with settings for the GPILC run. See `gpilc` function for details.
        U: np.ndarray
            Eigenvectors of the covariance matrix.
        λ: np.ndarray
            Eigenvalues of the covariance matrix.
        cov: np.ndarray
            Covariance matrix of the input maps.
            Shape should be (n_pixels, n_channels, n_channels) for 3D covariance or (n_channels, n_channels) for 2D covariance.
        cov_n: np.ndarray
            Covariance matrix of the nuisance maps.
            Shape should match cov.
        input_shapes: tuple
            Shapes of the input maps.
        compsep_run: dict   
            Dictionary with component separation parameters. See `gpilc` function for details.
        depro_cmb: float, optional
            Deprojection factor for CMB. If None, no deprojection is applied.
            Otherwise CMB residuals in GPILC maps will be at the level of depro_cmb * CMB_input.  
        m_bias: int or float, optional
            It will include m_bias more (if m_bias > 0) or less (if m_bias < 0) modes in the reconstructed GPILC maps.
    
    Returns
    -------
        np.ndarray
            Weights for GPILC. Shape is (n_pixels, n_channels, n_channels) for 3D covariance or (n_channels, n_channels) for 2D covariance.
    """  

    bandwidths = _get_bandwidths(config, compsep_run["good_channels"])
    A_cmb = _get_CMB_SED(np.array(config.instrument.frequency)[compsep_run["good_channels"]], 
                units=config.units, bandwidths=bandwidths) 

    if cov.ndim == 2:
        inv_cov = lg.inv(cov)

        m = _get_gilc_m(λ)
        m += int(m_bias)  

        U_s = np.delete(U,np.where(λ < λ[m-1]),axis=1)

        F = scipy.linalg.sqrtm(cov_n) @ U_s
        
        if depro_cmb is None:
            W = F @ lg.inv(F.T @ inv_cov @ F) @ F.T @ inv_cov
        else:
#            F_e = np.column_stack([F, depro_cmb * np.ones(input_shapes[0])])
            F_e = np.column_stack([F, depro_cmb * A_cmb])
            F_A = np.column_stack([F, A_cmb])
            W =  F_e @ lg.inv(F_A.T @ inv_cov @ F_A) @ F_A.T @ inv_cov

    elif cov.ndim == 3:
        m = _get_gilc_m(λ)
        m += int(m_bias)  

        covn_sqrt = lg.cholesky(cov_n)
        covn_sqrt_inv = lg.inv(covn_sqrt)

        W_=np.zeros((cov.shape[0],input_shapes[0],input_shapes[0]))

        for m_ in np.unique(m):
            U_s = U[m==m_,:,:m_]
            cov_inv = lg.inv(cov[m==m_])
            F = np.einsum("kij,kjz->kiz", covn_sqrt[m==m_],U_s)

            if depro_cmb is None: 
                W_[m==m_] = np.einsum("kil,klj->kij",F,np.einsum("kzl,klj->kzj",lg.inv(np.einsum("kiz,kij,kjl->kzl",F,cov_inv,F)),np.einsum("kiz,kij->kzj",F,cov_inv)))
            else:
#                e_cmb = depro_cmb * np.ones((F.shape[0],F.shape[1],1)) 
#                F_e = np.concatenate((F,e_cmb),axis=2)
                F_e = np.concatenate((F,np.tile(depro_cmb * A_cmb, (F.shape[0], 1))[:, :, np.newaxis]),axis=2)
                F_A = np.concatenate((F,np.tile(A_cmb, (F.shape[0], 1))[:, :, np.newaxis]),axis=2)
                W_[m==m_] = np.einsum("kil,klj->kij",F_e,np.einsum("kzl,klj->kzj",lg.inv(np.einsum("kiz,kij,kjl->kzl",F_A,cov_inv,F_A)),np.einsum("kiz,kij->kzj",F_A,cov_inv)))

        if "mask" not in compsep_run and W_.shape[0] != input_shapes[-2]:
            W = np.zeros((input_shapes[-2],W_.shape[1],W_.shape[2]))
            for i, k in np.ndindex(W_.shape[1],W_.shape[2]):
                W[:,i,k]=hp.ud_grade(W_[:,i,k],hp.npix2nside(input_shapes[-2]))
        else:
            W=np.copy(W_)

    return W

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]
                    


