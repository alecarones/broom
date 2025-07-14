import numpy as np
import healpy as hp
from .configurations import Configs
from .routines import _get_local_cov, _EB_to_QU, _E_to_QU, _B_to_QU, obj_to_array, array_to_obj, _get_bandwidths
from .saving import _save_compsep_products, _get_full_path_out, save_ilc_weights
from .needlets import _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering, _get_good_channels_nl
from .ilcs import _standardize_cilc, get_inv_cov
from .leakage import purify_master, purify_recycling
from .seds import _get_CMB_SED, _get_moments_SED, _standardize_cilc
from types import SimpleNamespace
import os
import re
from typing import Dict, Any, Union, Optional, Tuple, List
import sys

def pilc(config: Configs, input_alms: SimpleNamespace, compsep_run: Dict[str, Any], **kwargs) -> Union[SimpleNamespace, None]:
    """
    Perform Polarization Internal Linear Combination (PILC) on input alm coefficients.

    Parameters
    ----------
        config : Configs
            Configuration object containing global settings. It includes:
            - lmax : int, maximum multipole for the component separation.
            - nside : int, HEALPix resolution parameter for compsep products.
            - fwhm_out : float, full width at half maximum of the output beam in arcminutes.
            - pixel_window_out : bool, whether to apply a pixel window to the output maps.
            - field_out : str, desired output fields (e.g., "E", "B", "QU", "QU_E", "QU_B").
            - save_compsep_products : bool, whether to save component separation products.
            - return_compsep_products : bool, whether to return component separation products.
            - path_outputs : str, path to save the output files.
            - units : str, units of the output maps (e.g., "uK_CMB"). Used to compute moments, if needed.
            - bandpass_integrate : bool, whether inputs are bandpass-integrated. Used to compute moments, if needed. 

        input_alms : SimpleNamespace
            SimpleNamespace object containing input spherical harmonic coefficients (alms). 
            Each attribute has shape (n_channels, (n_fields), n_alms), where n_fields depend on the fields requested in config.field_out. 
            n_fields can be None if only E and B are requested, or 2 if both are requested to be included in P component separation.

        compsep_run : dict
            Dictionary specifying component separation parameters. It includes:
            - "method": Method to use for component separation (e.g., "pilc", "cpilc").
            - "domain": Domain of the component separation (e.g., "pixel", "needlet").
            - "needlet_config": Configuration for needlet parameters (if applicable).
            - "ilc_bias": Parameter setting the residual percentage ILC bias (default is 0. and covariance will be computed over the whole sky).
            - "reduce_ilc_bias": Boolean. If True, it implements a procedure to attenuate ILC bias.
            - "save_needlets": boolean. Whether to save adopted needlet bands in the output directory. 
                If not provided, it is set to the value of config.save_compsep_products. Default: True.
            - "save_weights": boolean. Whether to save compsep weights maps in the output directory.
                If not provided, it is set to the value of config.save_compsep_products. Default: True.
            - "mask": (optional) HEALPix mask which will exclude (unobserved) regions from covariance computation. 
                It must be a 1D array with shape (12 * nside**2,). If non-binary, it will be used to weigh pixel in covariance computation.
            - "cov_noise_debias": (float, list) Noise covariance debiasing factor. 
                If different from 0. it will debias the covariance matrix by a factor cov_noise_debias * noise_covariance.
                It must be a list with the same length as the number of needlet scales if domain is "needlet", otherwise a single float.
            - "special_nls": (list) List of needlet scales where moment deprojection is applied in c_pilc.
            - "constraints": Dictionary to be used if method is "cpilc" or "c_pilc". It must contain:
                - "moments": list of strings with moments to be deprojected in cPILC. It can be a list of lists if domain is "needlet".
                    Each list will be associated to a needlet band in order.
            Optionally can include:
                - "beta_d": (float, list): dust spectral index to compute dust moments. If list, a different beta_d is used for each needlet band. Default: 1.54.
                - "T_d": (float, list): dust temperature to compute dust moments. If list, a different T_d is used for each needlet band. Default: 19.6.
                - "beta_s": (float, list): synchrotron spectral index to compute synchrotron moments. If list, a different beta_s is used for each needlet band. Default: -3.
                - "deprojection": (float, list): deprojection factor for each moment. If list, a different deprojection factor is used for each moment.
                    It can be a list of lists if domain is "needlet". Each list will be associated to a needlet band in order. Default: 0. (i.e. full deprojection) for each moment and each needlet band.

        **kwargs : dict
            Additional keyword arguments forwarded to hp.map2alm.

    Returns
    -------
        SimpleNamespace or None
            Output maps if config.return_compsep_products is True, otherwise None.
    """

    if config.field_out not in ["E", "B", "QU", "EB", "QU_E", "QU_B"]:
        raise ValueError("Invalid field_out for PILC. It must be E, B, QU, EB, QU_E, or QU_B.")

    if compsep_run["method"] == "cpilc" or compsep_run["method"] == "c_pilc":
        compsep_run = _standardize_cilc(compsep_run, config.lmax)

    if np.any(np.array(compsep_run["cov_noise_debias"]) != 0.):
        if not hasattr(input_alms, "noise"):
            raise ValueError("The input_alms object must have 'noise'' attribute for debiasing the covariance.")
        compsep_run["noise_idx"] = 2 if hasattr(input_alms, "fgds") else 1

    output_maps = _pilc(config, obj_to_array(input_alms), compsep_run, **kwargs)
    
    outputs = array_to_obj(output_maps, input_alms)

    del output_maps

    if config.save_compsep_products:
        _save_compsep_products(config, outputs, compsep_run, nsim=compsep_run["nsim"])
    if config.return_compsep_products:
        return outputs
    return None

def _pilc(config: Configs, input_alms: np.ndarray, compsep_run: Dict[str, Any], **kwargs) -> np.ndarray:
    """
    Internal PILC dispatcher to either pixel or needlet domain implementation.

    Parameters
    ----------
        config : Configs
            Configuration object containing global settings. See 'pilc' function for details.
        input_alms : np.ndarray
            Input spherical harmonic coefficients. 
            It must have shape (n_channels, n_fields, n_alms, n_components) if both E- and B-modes are provided, (n_channels, n_alms, n_components) otherwise.
        compsep_run : dict
            Component separation configuration dictionary. See 'pilc' function for details.
        **kwargs : dict
            Additional keyword arguments for hp.map2alm.

    Returns
    -------
        np.ndarray
            Output maps from PILC.
    """
    if input_alms.ndim == 4:
        if input_alms.shape[1] != 2:
            raise ValueError("input_alms must have shape (nfreq, 2, nalm, ncomps) for pilc.")
        compsep_run["field"] = "QU"

    elif input_alms.ndim == 3:
        if config.field_out in ["E", "QU_E"]:
            compsep_run["field"] = "QU_E"
        elif config.field_out in ["B", "QU_B"]:
            compsep_run["field"] = "QU_B"

    if compsep_run["domain"] == "pixel":
        output_maps = _pilc_pixel(config, input_alms, compsep_run, **kwargs)
    elif compsep_run["domain"] == "needlet":
        output_maps = _pilc_needlet(config, input_alms, compsep_run, **kwargs)

    del compsep_run["field"]

    return output_maps

def _pilc_needlet(config: Configs, input_alms: np.ndarray, compsep_run: Dict[str, Any], **kwargs) -> np.ndarray:
    """
    Perform PILC in the needlet domain.

    Parameters
    ----------
        config : Configs
            Configuration object with global settings. See 'pilc' function for details.
        input_alms : np.ndarray
            Input alm array. Shape can be:
            - (n_channels, 2, n_alms, n_components) for EB inputs
            - (n_channels, n_alms, n_components) for E or B inputs
        compsep_run : dict
            Component separation parameters including needlet configuration. See 'pilc' function for details.
        **kwargs : dict
            Additional keyword arguments for hp.map2alm.

    Returns
    -------
        np.ndarray
            Output maps after application of needlet domain PILC.
    """
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    b_ell = b_ell**2

    if compsep_run['save_needlets']:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        os.makedirs(compsep_run["path_out"], exist_ok=True)
        np.save(os.path.join(compsep_run["path_out"], "needlet_bands"), b_ell)
    
    output_maps = np.zeros((2, hp.nside2npix(config.nside), input_alms.shape[-1]))
    
    for j in range(b_ell.shape[0]):
        output_maps += _pilc_needlet_j(config, input_alms, compsep_run, b_ell[j], j)
    
    if ((config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out) or \
       (config.field_out in ["E", "B", "EB"]):

        for c in range(output_maps.shape[-1]):
#            if "mask" in compsep_run and config.mask_type == "observed_patch" and config.leakage_correction is not None:
#                if "_purify" in config.leakage_correction:
#                    alm_out = purify_master(output_maps[...,c], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction))
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
#                elif "_recycling" in config.leakage_correction:
#                    if "_iterations" in config.leakage_correction:
#                        iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
#                    else:
#                        iterations = 0
#                    alm_out = purify_recycling(output_maps[...,c], output_maps[...,0], np.ceil(compsep_run["mask"]), config.lmax,
#                                            purify_E=("E" in config.leakage_correction),
#                                            iterations=iterations)
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
#                elif config.leakage_correction=="mask_only":
#                    alm_out = hp.map2alm(np.array([0.*output_maps[0,:,c],output_maps[0,:,c],output_maps[1,:,c]]) * compsep_run["mask"],
#                                         lmax=config.lmax, pol=True, **kwargs)
#            else:
            alm_out = hp.map2alm([0.*output_maps[0,:,c],output_maps[0,:,c],output_maps[1,:,c]],
                                         lmax=config.lmax, pol=True, **kwargs)

            if (config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out:
                output_maps[...,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=True, pixwin=True)[1:]
            elif config.field_out in ["E","B"]:
                output_maps[0,:,c] = hp.alm2map(alm_out[1] if config.field_out=="E" else alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
            elif config.field_out=="EB":
                output_maps[0,:,c] = hp.alm2map(alm_out[1], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
                output_maps[1,:,c] = hp.alm2map(alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)

        if config.field_out in ["E","B"]:
            output_maps = output_maps[0]

    if "mask" in compsep_run:
        if output_maps.ndim == 3:
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        elif output_maps.ndim == 2:
            output_maps[compsep_run["mask"] == 0.,:] = 0.

    return output_maps

def _pilc_needlet_j(config: Configs, input_alms: np.ndarray, compsep_run: Dict[str, Any], b_ell: np.ndarray, nl_scale: int) -> np.ndarray:
    """
    Compute the needlet-scale PILC component separated maps.

    Parameters
    ----------
        config : Configs
            Configuration object. See 'pilc' function for details.
        input_alms : np.ndarray
            Input alm array. It can have shape:
            - (n_channels, 2, n_alms, n_comps) for EB inputs
            - (n_channels, n_alms, n_comps) for E or B inputs
        compsep_run : dict
            Component separation configuration. See 'pilc' function for details.
        b_ell : np.ndarray
            Needlet window function. Should be a 1D array of shape (lmax+1,).
        nl_scale : int
            Current needlet scale index.
    
    Returns
    -------
        np.ndarray
            Output alm array at the requested needlet scale.
    """    
    nside_, lmax_ = config.nside, config.lmax

    good_channels_nl = _get_good_channels_nl(config, b_ell)

    input_maps_nl = np.zeros((good_channels_nl.shape[0], 2, 12 * nside_**2, input_alms.shape[-1]))
    for n, channel in enumerate(good_channels_nl):
        input_alms_j = np.zeros((2, hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)
        if input_alms.ndim == 4:
            for k in range(2):
                input_alms_j[k] = _needlet_filtering(input_alms[channel,k], b_ell, lmax_)
        elif input_alms.ndim == 3:
            if config.field_out in ["QU_E", "E"]:
                input_alms_j[0] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
            elif config.field_out in ["QU_B", "B"]:
                input_alms_j[1] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        
        input_alms_j = np.ascontiguousarray(input_alms_j)

        for c in range(input_alms.shape[-1]):
            input_maps_nl[n,...,c] = hp.alm2map(np.array([0.*input_alms_j[0, :, c],input_alms_j[0, :, c],input_alms_j[1, :, c]]), nside_, lmax=lmax_, pol=True)[1:]   

    output_maps_nl = _pilc_maps(config, input_maps_nl, compsep_run, b_ell, nl_scale=nl_scale)
    
    del input_maps_nl

    return output_maps_nl

def _pilc_pixel(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: Dict,
    **kwargs
) -> np.ndarray:
    """
    Perform PILC in the pixel domain.

    Parameters:
    ----------
        config : Configs
            Configuration object containing global parameters (nside, lmax, field_out, etc.). See 'pilc' function for details.
        input_alms : np.ndarray
            Input harmonic coefficients. Shape can be:
            - (n_freqs, 2, n_alm, n_comps) for EB inputs
            - (n_freqs, n_alm, n_comps) for E or B inputs
        compsep_run : dict
            Dictionary containing component separation parameters (e.g., method, mask, etc.). See 'pilc' function for details.
        **kwargs
            Additional arguments passed to healpy map2alm.

    Returns
    -------
        np.ndarray
            Component-separated output maps. Shape depends on `config.field_out`:
            - For "QU" or "EB, shape is (2, npix, n_comps)
            - For "E" or "B", shape is (npix, n_comps)
    """
    good_channels = _get_good_channels_nl(config, np.ones(config.lmax+1))

    input_maps = np.zeros((good_channels.shape[0], 2, 12 * config.nside**2, input_alms.shape[-1]))

    for n, channel in enumerate(good_channels):
        for c in range(input_alms.shape[-1]):
            if input_alms.ndim == 4:
                input_maps[n,...,c] = hp.alm2map(np.array([0.*input_alms[channel, 0, :, c],input_alms[channel, 0, :, c],input_alms[channel, 1, :, c]]), config.nside, lmax=config.lmax, pol=True)[1:]      
            elif input_alms.ndim == 3:
                if config.field_out in ["QU_E", "E"]:
                    input_maps[n,...,c] = hp.alm2map(np.array([0.*input_alms[channel, :, c],input_alms[channel, :, c],0.*input_alms[channel, :, c]]), config.nside, lmax=config.lmax, pol=True)[1:]
                elif config.field_out in ["QU_B", "B"]:
                    input_maps[n,...,c] = hp.alm2map(np.array([0.*input_alms[channel, :, c],0.*input_alms[channel, :, c],input_alms[channel, :, c]]), config.nside, lmax=config.lmax, pol=True)[1:]

    output_maps = _pilc_maps(config, input_maps, compsep_run, np.ones(config.lmax+1))

    if ((config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out) or \
       (config.field_out in ["E", "B", "EB"]):

        for c in range(output_maps.shape[-1]):
#            if "mask" in compsep_run and config.mask_type == "observed_patch" and config.leakage_correction is not None:
#                if "_purify" in config.leakage_correction:
#                    alm_out = purify_master(output_maps[...,c], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction))
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
#                elif "_recycling" in config.leakage_correction:
#                    if "_iterations" in config.leakage_correction:
#                        iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
#                    else:
#                        iterations = 0
#                    alm_out = purify_recycling(output_maps[...,c], output_maps[...,0], np.ceil(compsep_run["mask"]), config.lmax,
#                                            purify_E=("E" in config.leakage_correction),
#                                            iterations=iterations)
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
#                elif config.leakage_correction=="mask_only":
#                    alm_out = hp.map2alm(np.array([0.*output_maps[0,:,c],output_maps[0,:,c],output_maps[1,:,c]]) * compsep_run["mask"],
#                                         lmax=config.lmax, pol=True, **kwargs)
#            else:
            alm_out = hp.map2alm([0.*output_maps[0,:,c],output_maps[0,:,c],output_maps[1,:,c]],
                                        lmax=config.lmax, pol=True, **kwargs)

            # Convert Alms back to maps
            if (config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out:
                output_maps[...,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=True, pixwin=True)[1:]
            elif config.field_out in ["E","B"]:
                idx = 1 if config.field_out == "E" else 2                
                output_maps[0, :, c] = hp.alm2map(alm_out[idx], config.nside, lmax=config.lmax, pol=False,
                                                  pixwin=config.pixel_window_out)
            elif config.field_out=="EB":
                output_maps[0, :, c] = hp.alm2map(alm_out[1], config.nside, lmax=config.lmax, pol=False,
                                                  pixwin=config.pixel_window_out)
                output_maps[1, :, c] = hp.alm2map(alm_out[2], config.nside, lmax=config.lmax, pol=False,
                                                  pixwin=config.pixel_window_out)

        # Reduce shape if only E or B was requested
        if config.field_out in ["E","B"]:
            output_maps = output_maps[0]

    # Apply mask again 
    if "mask" in compsep_run:
        if output_maps.ndim == 3:
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        elif output_maps.ndim == 2:
            output_maps[compsep_run["mask"] == 0.,:] = 0.

    return output_maps
   
def _pilc_maps(
    config: Configs,
    input_maps: np.ndarray,
    compsep_run: Dict,
    b_ell: np.ndarray,
    nl_scale: Optional[Union[int, None]] = None
) -> np.ndarray:
    """
    Performs Polarization ILC (PILC) map reconstruction.

    Parameters:
    ----------
        config: Configs
            Configuration object containing instrument and analysis settings. See 'pilc' function for details.
        input_maps: np.ndarray
            Input maps array of shape (n_channels, 2, n_pix, n_comps), where the second dimension corresponds to QU polarization maps,
            while n_comp corresponds to the number of components (e.g., total, CMB, foregrounds, noise).
        compsep_run: Dict
            Dictionary containing component separation parameters and settings. See 'pilc' function for details.
        b_ell: np.ndarray
            Needlet window function for the current scale. It should be a 1D array of shape (lmax+1,).
            If the method domain is "pixel", it should be an array of ones with shape (lmax+1,).
        nl_scale: int, optional
            Needlet scale index for which the PILC is performed. It has to be not None if the method domain is "needlet".

    Returns
    -------
        output_maps: np.ndarray
            Reconstructed PILC CMB maps of shape (2, n_pix, n_comps), with 2 corresponding to QU fields.
    """

    good_channels = _get_good_channels_nl(config, b_ell)
    freqs = np.array(config.instrument.frequency)[good_channels]

    bandwidths = _get_bandwidths(config, good_channels)

    A_cmb = _get_CMB_SED(freqs, units=config.units, bandwidths=bandwidths)

    def _assign_moment_constraints(scale_idx=None):
        idx = scale_idx if scale_idx is not None else slice(None)
        compsep_run["A"] = _get_moments_SED(
            freqs,
            compsep_run["constraints"]["moments"][idx],
            beta_d=compsep_run["constraints"]["beta_d"][idx],
            T_d=compsep_run["constraints"]["T_d"][idx],
            beta_s=compsep_run["constraints"]["beta_s"][idx],
            units=config.units,
            bandwidths=bandwidths
        )
        compsep_run["e"] = np.array(compsep_run["constraints"]["deprojection"][idx])

    if compsep_run["method"] == "cpilc":
        _assign_moment_constraints(nl_scale)
    elif compsep_run["method"] == "c_pilc" and nl_scale in compsep_run.get("special_nls", []):
        scale_idx = compsep_run["special_nls"] == nl_scale
        _assign_moment_constraints(scale_idx)

    cov = get_prilc_cov(input_maps[...,0], config.lmax, compsep_run, b_ell)
    noise_debias = compsep_run["cov_noise_debias"] if compsep_run["domain"] == "pixel" else compsep_run["cov_noise_debias"][nl_scale]
    if noise_debias != 0.:
        cov_n = get_prilc_cov(input_maps[...,compsep_run["noise_idx"]], config.lmax, compsep_run, b_ell)
        cov = cov - noise_debias * cov_n
        del cov_n

    inv_cov = get_inv_cov(cov)
    del cov

    w_pilc = get_pilc_weights(A_cmb, inv_cov, input_maps.shape, compsep_run)
    del inv_cov

    if compsep_run["save_weights"]:
        #if 'path_out' not in compsep_run:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        save_ilc_weights(config, w_pilc, compsep_run,
                         hp.npix2nside(input_maps.shape[-2]), nl_scale=nl_scale)
        
    compsep_run.pop("A", None)
    compsep_run.pop("e", None)

    if compsep_run["ilc_bias"] == 0.:
        if w_pilc.ndim==1:
            output_maps = np.einsum('i,ifjk->fjk', w_pilc, input_maps)
        elif w_pilc.ndim==2:
            Q = np.einsum('i,ijk->jk', w_pilc[0], input_maps[:, 0])
            U = np.einsum('i,ijk->jk', w_pilc[1], input_maps[:, 0])
            Q -= np.einsum('i,ijk->jk', w_pilc[1], input_maps[:, 1])
            U += np.einsum('i,ijk->jk', w_pilc[0], input_maps[:, 1])
            output_maps = np.stack([Q, U])
    else:
        if w_pilc.ndim==2:
            output_maps = np.einsum('ij,ifjk->fjk', w_pilc, input_maps)
        elif w_pilc.ndim==3:
            Q = np.einsum('ij,ijk->jk', w_pilc[0], input_maps[:, 0])
            U = np.einsum('ij,ijk->jk', w_pilc[1], input_maps[:, 0])
            Q -= np.einsum('ij,ijk->jk', w_pilc[1], input_maps[:, 1])
            U += np.einsum('ij,ijk->jk', w_pilc[0], input_maps[:, 1])
            output_maps = np.stack([Q, U])
            
    return output_maps

def get_pilc_weights(
    A_cmb: np.ndarray,
    inv_cov: np.ndarray,
    input_shapes: Tuple[int, ...],
    compsep_run: Dict
) -> np.ndarray:
    """
    Compute weights for the Polarization Internal Linear Combination (PILC) method,
    with or without constraints to deproject additional components.

    Parameters:
    ----------
        A_cmb: np.ndarray
            Spectral energy distribution (SED) of the CMB. Shape: (n_channels,)
        inv_cov: np.ndarray
            Inverse covariance matrix of input polarization maps:
            - If ilc_bias=0: shape (n_channels, n_channels)
            - If ilc_bias!=0: shape (n_channels, n_channels, n_pixels)
        input_shapes: Tuple[int, ...]
            Shape of the input map (n_channels, 2, n_pixels, n_comps) or equivalent, used to determine spatial size and channels.
        compsep_run: Dict
            Dictionary with component separation settings. See 'pilc' function for details.

    Returns
    -------
        w_ilc: np.ndarray
            ILC weights, shaped:
                - (n_channels,) if ilc_bias = 0
                - (n_channels, n_pixels) if ilc_bias != 0
    """

    if "A" in compsep_run:
        # Add CMB constraint at the top of the A matrix and 'e' constraint vector
        compsep_run["A"] = np.vstack((A_cmb, compsep_run["A"]))
        compsep_run["e"] = np.insert(compsep_run["e"], 0, 1.)

        if compsep_run["ilc_bias"] == 0.:
            # Spatially-invariant weights (simpler case)
            inv_ACA = np.linalg.inv(
                np.einsum("zi,il->zl", compsep_run["A"], np.einsum("ij,lj->il", inv_cov, compsep_run["A"]))
            )
            w_ilc = np.einsum(
                "l,lj->j",
                compsep_run["e"],
                np.einsum("lz,zj->lj", inv_ACA, np.einsum("zi,ij->zj", compsep_run["A"], inv_cov))
            )
            del inv_ACA
        else:
            # Spatially-varying weights
            inv_ACA = np.linalg.inv(
                np.einsum("zi,ilk->zlk", compsep_run["A"], np.einsum("ijk,lj->ilk", inv_cov, compsep_run["A"])).T).T

            w_ilc=np.zeros((input_shapes[0],input_shapes[-2]))
            w_ = np.einsum(
                "l,ljk->jk",
                compsep_run["e"],
                np.einsum("lzk,zjk->ljk", inv_ACA, np.einsum("zi,ijk->zjk", compsep_run["A"], inv_cov))
            )

            for i in range(input_shapes[0]):
                if "mask" in compsep_run:
                    w_ilc[i] = w_[i]
                else:
                    w_ilc[i]=hp.ud_grade(w_[i],hp.npix2nside(input_shapes[-2]))
            del w_, inv_ACA
    else:
        if compsep_run["ilc_bias"] == 0.:
            w_ilc = (A_cmb.T @ inv_cov) / (A_cmb.T @ inv_cov @ A_cmb) 
        else:
            w_ilc=np.zeros((input_shapes[0],input_shapes[-2]))
            AT_invC = np.einsum('j,ijk->ik', A_cmb, inv_cov) # np.sum(inv_cov,axis=1)
            AT_invC_A = np.einsum('j,ijk, i->k', A_cmb, inv_cov, A_cmb) #np.sum(inv_cov,axis=(0,1))
            for i in range(input_shapes[0]):
                if "mask" in compsep_run:
                    w_ilc[i] = AT_invC[i]/AT_invC_A
                else:
                    w_ilc[i]=hp.ud_grade(AT_invC[i]/AT_invC_A,hp.npix2nside(input_shapes[-2]))
    return w_ilc


def get_pilc_cov(
    input_maps: np.ndarray,
    lmax: int,
    compsep_run: Dict,
    b_ell: np.ndarray
) -> np.ndarray:
    """
    Compute full PILC covariance matrix.

    Parameters:
    ----------
        input_maps: np.ndarray
            input Q/U maps. Shape (n_channels, 2, n_pixels)
        lmax: int
            Maximum multipole for the analysis.
        compsep_run: Dict
            Dictionary with component separation settings. See 'pilc' function for details.
        b_ell: np.ndarray
            Needlet window function for the current scale. It should be a 1D array of shape (lmax+1,).

    Returns
    -------
        cov: np.ndarray
            Full 2x2 polarization covariance matrix of shape (2*n_channels, 2*n_channels, [n_pixels]).
    """

    if compsep_run["ilc_bias"] == 0.:
        if "mask" in compsep_run:
            mask = compsep_run["mask"] > 0.
            cov_qq_uu = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0, mask], input_maps[:, 0, mask]), axis=-1) + \
                        np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1, mask], input_maps[:, 1, mask]), axis=-1)
            cov_qu_uq = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0, mask], input_maps[:, 1, mask]), axis=-1) - \
                        np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1, mask], input_maps[:, 0, mask]), axis=-1)
        else:
            cov_qq_uu = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0], input_maps[:, 0]), axis=-1) + \
                        np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1], input_maps[:, 1]), axis=-1)
            cov_qu_uq = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0], input_maps[:, 1]), axis=-1) - \
                        np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1], input_maps[:, 0]), axis=-1)
    else:
        mask = compsep_run.get("mask")
        reduce_bias = compsep_run["reduce_ilc_bias"]

        cov_qq_uu = _get_local_cov(input_maps[:, 0], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=mask, reduce_bias=reduce_bias) + \
                    _get_local_cov(input_maps[:, 1], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=mask, reduce_bias=reduce_bias)
        cov_qu_uq = _get_local_cov(input_maps[:, 0], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=mask, reduce_bias=reduce_bias, input_maps_2=input_maps[:, 1]) - \
                    _get_local_cov(input_maps[:, 1], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=mask, reduce_bias=reduce_bias, input_maps_2=input_maps[:, 0])

        if mask is not None and cov_qq_uu.shape[-1] == input_maps.shape[-1]:
            valid = mask > 0.0
            fallback_cov_qq_uu = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0, valid], input_maps[:, 0, valid]), axis=-1) + \
                           np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1, valid], input_maps[:, 1, valid]), axis=-1)
            cov_qq_uu[..., mask == 0.0] = fallback_cov_qq_uu
            fallback_cov_qu_uq = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0, valid], input_maps[:, 1, valid]), axis=-1) - \
                           np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1, valid], input_maps[:, 0, valid]), axis=-1)
            cov_qu_uq[..., mask == 0.0] = np.repeat(fallback_cov_qu_uq[..., np.newaxis], np.sum(mask == 0.0), axis=-1)
    
    cov = np.concatenate((
        np.concatenate((cov_qq_uu, -cov_qu_uq), axis=1),
        np.concatenate((cov_qu_uq, cov_qq_uu), axis=1)
    ), axis=0)

    return cov

def get_prilc_cov(
    input_maps: np.ndarray,
    lmax: int,
    compsep_run: Dict,
    b_ell: np.ndarray
) -> np.ndarray:
    """
    Compute the reduced ILC covariance matrix for polarization maps.

    Parameters:
    ----------
        input_maps: np.ndarray
            input Q/U maps. Shape (n_channels, 2, n_pixels)
        lmax: int
            Maximum multipole for the analysis.
        compsep_run: Dict
            Dictionary with component separation settings. See 'pilc' function for details.
        b_ell: np.ndarray
            Needlet window function for the current scale. It should be a 1D array of shape (lmax+1,).

    Returns
    -------
        cov: np.ndarray
            Covariance matrix of shape (n_channels, n_channels, [n_pixels]).
    """
    if compsep_run["ilc_bias"] == 0.:
        if "mask" in compsep_run:
            mask = compsep_run["mask"] > 0.0
            cov = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0, mask], input_maps[:, 0, mask]), axis=-1) + \
                  np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1, mask], input_maps[:, 1, mask]), axis=-1)
        else:
            cov = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0], input_maps[:, 0]), axis=-1) + \
                  np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1], input_maps[:, 1]), axis=-1)

    else:
        mask = compsep_run.get("mask")
        reduce_bias = compsep_run["reduce_ilc_bias"]

        cov = _get_local_cov(input_maps[:, 0], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=mask, reduce_bias=reduce_bias) + \
              _get_local_cov(input_maps[:, 1], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=mask, reduce_bias=reduce_bias)

        if mask is not None and cov.shape[-1] == input_maps.shape[-1]:
            mask_valid = mask > 0.0
            fallback_cov = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0, mask_valid], input_maps[:, 0, mask_valid]), axis=-1) + \
                           np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1, mask_valid], input_maps[:, 1, mask_valid]), axis=-1)
            cov[..., mask == 0.0] = np.repeat(fallback_cov[..., np.newaxis], np.sum(mask == 0.0), axis=-1)

    return cov

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]
                    


