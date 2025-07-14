import numpy as np
import healpy as hp
from .configurations import Configs
from .routines import _get_local_cov, _EB_to_QU, _E_to_QU, _B_to_QU,\
                      obj_to_array, array_to_obj, _get_bandwidths
from .saving import _save_compsep_products, _get_full_path_out, save_patches, save_ilc_weights
from .needlets import  _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering, _get_good_channels_nl 
from .seds import _get_CMB_SED, _get_moments_SED, _standardize_cilc
from .clusters import _adapt_tracers_path, _cea_partition, _rp_partition, \
                      get_scalar_tracer, get_scalar_tracer_nl, initialize_scalar_tracers

from types import SimpleNamespace
import os
from typing import Any, Optional, Union, Dict, List, Tuple
import sys

def ilc(config: Configs, input_alms: SimpleNamespace, compsep_run: Dict, **kwargs: Any) -> Optional[SimpleNamespace]:
    """
    Perform Internal Linear Combination (ILC) component separation (all its variants) on CMB scalar fields.

    Parameters
    ----------
        config : Configs
            Configuration object containing global settings. It includes:
            - lmax : int, maximum multipole for the component separation.
            - nside : int, HEALPix resolution parameter for compsep products.
            - fwhm_out : float, full width at half maximum of the output beam in arcminutes.
            - pixel_window_out : bool, whether to apply a pixel window to the output maps.
            - field_out : str, desired output fields (e.g., "T", "E", "B", "QU", "TQU", "QU_E", "QU_B").
            - save_compsep_products : bool, whether to save component separation products.
            - return_compsep_products : bool, whether to return component separation products.
            - path_outputs : str, path to save the output files.
            - units : str, units of the output maps (e.g., "uK_CMB"). Used to compute moments, if needed.
            - bandpass_integrate : bool, whether inputs are bandpass-integrated. Used to compute moments, if needed. 
        input_alms : SimpleNamespace
            SimpleNamespace object containing input spherical harmonic coefficients (alms). 
            Each attribute has shape (n_channels, (n_fields), n_alms), where n_fields depend on the fields requested in config.field_out.
        compsep_run : dict
            Dictionary specifying component separation parameters. It includes:
            - "method": Method to use for component separation (e.g., "ilc", "cilc", "mc_ilc").
            - "domain": Domain of the component separation (e.g., "pixel", "needlet").
            - "needlet_config": Dictionary containing needlet settings:
                - "needlet_windows": Type of needlet windows ('cosine', 'standard', 'mexican').
                - "ell_peaks": List of integers defining multipoles of peaks for needlet bands (required for 'cosine').
                - "width": Width of the needlet windows (required for 'standard' and 'mexican').
                - "merging_needlets": Integer or list of integers defining ranges of needlets to be merged.
            - "ilc_bias": Parameter setting the residual percentage ILC bias (default is 0. and covariance will be computed over the whole sky).
            - "reduce_ilc_bias": Boolean. If True, it implements a procedure to attenuate ILC bias.
            - "b_squared": Boolean indicating if the needlet windows should be squared (if domain is "needlet").
            - "adapt_nside": Adapt HEALPix resolution of needlet maps to sampled multipole range. Deafult: False.
            - "save_needlets": boolean. Whether to save adopted needlet bands in the output directory. 
                If not provided, it is set to the value of config.save_compsep_products. Default: True.
            - "save_weights": boolean. Whether to save compsep weights maps in the output directory.
                If not provided, it is set to the value of config.save_compsep_products. Default: True.
            - "mask": (optional) HEALPix mask which will exclude (unobserved) regions from covariance computation. 
                It must be a 1D array with shape (12 * nside**2,). If non-binary, it will be used to weigh pixel in covariance computation.
            - "n_patches": (int) Number of patches to use in MC-ILC. Default: 50.
            - "mc_type": (str), needed if method is "mcilc" or "mc_ilc".
                Type of MC-ILC to use. Options: "cea_ideal", "rp_ideal", "cea_real", "rp_real". Default: "cea_real".
            - "cov_noise_debias": (float, list) Noise covariance debiasing factor. 
                If different from 0. it will debias the covariance matrix by a factor cov_noise_debias * noise_covariance.
                It must be a list with the same length as the number of needlet scales if domain is "needlet", otherwise a single float.
            - "special_nls": (list) List of needlet scales where moment deprojection is applied in c_ilc and clustering in mc_ilc.
            - "constraints": Dictionary to be used if method is "cilc" or "c_ilc". It must contain:
                - "moments": list of strings with moments to be deprojected in cILC. It can be a list of lists if domain is "needlet".
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
        outputs : SimpleNamespace or None
            Output separated components as a SimpleNamespace object if 
            config.return_compsep_products is True; otherwise None.
    """
    # Standardize cilc method parameters if needed
    if compsep_run["method"] in ["cilc", "c_ilc", "mc_cilc"]:
        compsep_run = _standardize_cilc(compsep_run, config.lmax)

    # Check for MC-ILC ideal tracer requirements
    if compsep_run["method"] in ["mcilc", "mc_ilc", "mc_cilc"]:
        if compsep_run["mc_type"] in ["cea_ideal", "rp_ideal"]:
            if not hasattr(input_alms, "fgds"):
                raise ValueError("The input_alms object must have 'fgds' attribute for ideal tracer in MC-ILC.")
    
    if compsep_run["method"] != "mcilc":
        if np.any(np.array(compsep_run["cov_noise_debias"]) != 0.):
            if not hasattr(input_alms, "noise"):
                raise ValueError("The input_alms object must have 'noise'' attribute for debiasing the covariance.")
            compsep_run["noise_idx"] = 2 if hasattr(input_alms, "fgds") else 1
                    
    # Perform the core ILC component separation
    output_maps = _ilc(config, obj_to_array(input_alms), compsep_run, **kwargs)

    compsep_run.pop("noise_idx", None)
    
    # Post-process outputs (e.g., convert EB to QU if needed)
    output_maps = _ilc_post_processing(config, output_maps, compsep_run, **kwargs)

    # Convert array back to object with attributes
    outputs = array_to_obj(output_maps, input_alms)

    del output_maps

    # Save component separation products if requested
    if config.save_compsep_products:
        _save_compsep_products(config, outputs, compsep_run, nsim=compsep_run["nsim"])

    if config.return_compsep_products:
        return outputs

def _ilc_post_processing(
    config: Configs,
    output_maps: np.ndarray,
    compsep_run: Dict,
    **kwargs: Any
) -> np.ndarray:
    """
    Post-process the ILC output maps, including polarization conversions and masking.

    Parameters
    ----------
        config : Configs
            Configuration object containing output settings. See 'ilc' for details.
        output_maps : np.ndarray
            Output maps from the ILC process. 
            Shape is (n_fields, npix, n_components) where n_fields depends on the requested output field in config.field_out.
        compsep_run : dict
            Dictionary specifying component separation parameters. It requires:
            - "mask": (optional) HEALPix mask which will exclude (unobserved) regions from the output maps.
        **kwargs : dict
            Additional keyword arguments forwarded to hp.map2alm.

    Returns
    -------
        np.ndarray
            Post-processed output maps.
    """
    # Handle EB to QU conversion for 3D arrays with appropriate fields
    if output_maps.ndim == 3 and (
        (output_maps.shape[0] == 2 and config.field_out == "QU") or
        (output_maps.shape[0] == 3 and config.field_out == "TQU")
    ):
        outputs = np.zeros_like(output_maps)
        for c in range(output_maps.shape[-1]):
            outputs[:,:,c] = _EB_to_QU(output_maps[:,:,c],config.lmax, **kwargs)
        if "mask" in compsep_run:
            outputs[:,compsep_run["mask"] == 0.,:] = 0.
        return outputs
    # Handle QU_E and QU_B fields for 2D arrays
    elif (output_maps.ndim==2) and (config.field_out in ["QU_E", "QU_B"]):
        output = np.zeros((2, output_maps.shape[0], output_maps.shape[-1]))
        for c in range(output_maps.shape[-1]):
            if config.field_out == "QU_E":
                output[:,:,c] = _E_to_QU(output_maps[:,c],config.lmax, **kwargs)
            elif config.field_out=="QU_B":
                output[:,:,c] = _B_to_QU(output_maps[:,c],config.lmax, **kwargs)
        if "mask" in compsep_run:
            output[:,compsep_run["mask"] == 0.,:] = 0.
        return output
    # For other cases, apply mask if present
    else:
        if "mask" in compsep_run:
            if output_maps.ndim == 3:
                output_maps[:, compsep_run["mask"] == 0., :] = 0.
            elif output_maps.ndim == 2:
                output_maps[compsep_run["mask"] == 0.,:] = 0.
        return output_maps

def _ilc(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: Dict,
    **kwargs: Any
) -> np.ndarray:
    """
    Perform ILC on input alm arrays, either scalar or multi-field.

    Parameters
    ----------
        config : Configs
            Configuration object containing global settings. See 'ilc' for details.
        input_alms : np.ndarray
            Input spherical harmonic coefficients. 
            It must have shape (n_channels, n_fields, n_alms, n_components) for multifield or 
            (n_channels, n_alms, n_components) for scalar fields.
        compsep_run : dict
            Component separation configuration dictionary. See 'ilc' for details.
        **kwargs : dict
            Additional keyword arguments for hp.map2alm.

    Returns
    -------
        np.ndarray
            Output separated maps.
    """
    # Determine fields based on input shape and desired output
    if input_alms.ndim == 4:
        if input_alms.shape[1] == 3:
            fields_ilc = ["T", "E", "B"]
        elif input_alms.shape[1] == 2:
            fields_ilc = ["E", "B"]
    elif input_alms.ndim == 3:
        if config.field_out in ["T", "E", "B"]:
            fields_ilc = [config.field_out]
        elif config.field_out in ["QU_E", "QU_B"]:
            fields_ilc = [config.field_out[-1]]

    # Allocate output array and run ILC compsep per field if needed
    if input_alms.ndim == 4:
        output_maps = np.zeros((input_alms.shape[1], 12 * config.nside**2, input_alms.shape[3]))
        for i in range(input_alms.shape[1]):
            compsep_run["field"] = fields_ilc[i]
            if compsep_run["method"] in ["mcilc", "mc_ilc", "mc_cilc"]:
                input_fgds_alms = np.zeros_like(input_alms[:, i, :, 0]) if "real" in compsep_run["mc_type"] else input_alms[:, i, :, 1]
                compsep_run["tracers"] = initialize_scalar_tracers(
                    config, input_fgds_alms, compsep_run, field=compsep_run["field"], **kwargs
                )
                del input_fgds_alms
            output_maps[i] = _ilc_scalar(config, input_alms[:, i, :, :], compsep_run, **kwargs)

    elif input_alms.ndim == 3:
        compsep_run["field"] = fields_ilc[0]
        if compsep_run["method"] in ["mcilc", "mc_ilc", "mc_cilc"]:
            input_fgds_alms = np.zeros_like(input_alms[...,0]) if "real" in compsep_run["mc_type"] else input_alms[...,1]
            compsep_run["tracers"] = initialize_scalar_tracers(config, input_fgds_alms, compsep_run, field=compsep_run["field"], **kwargs)
            del input_fgds_alms
        output_maps = _ilc_scalar(config, input_alms, compsep_run, **kwargs)
    
    del compsep_run["field"]

    return output_maps

def _ilc_scalar(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: Dict,
    **kwargs: Any
) -> np.ndarray:
    """
    Execute ILC in the specified domain (pixel, needlet) on scalar field maps.

    Parameters
    ----------
        config : Configs
            Configuration object with global parameters. See 'ilc' for details.
        input_alms : np.ndarray
            Input alm array for a single scalar field. It must have shape (n_channels, n_alms, n_components).
        compsep_run : dict
            Component separation parameters. See 'ilc' for details.
        **kwargs : dict
            Additional keyword arguments passed internally.

    Returns
    -------
        np.ndarray
            Output maps after ILC processing.
    """
    if compsep_run["domain"] == "pixel":
        return _ilc_pixel(config, input_alms, compsep_run, **kwargs)
    elif compsep_run["domain"] == "needlet":
        return _ilc_needlet(config, input_alms, compsep_run, **kwargs)
#    elif compsep_run["domain"] == "harmonic":
#        output_maps = _hilc(config, input_alms, compsep_run)
    else:
        raise ValueError(f"Unsupported domain {compsep_run['domain']} in compsep.")

def _ilc_needlet(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: Dict,
    **kwargs: Any
) -> np.ndarray:
    """
    Perform ILC in the needlet domain.

    Parameters
    ----------
        config : Configs
            Configuration object with global settings. See 'ilc' for details.
        input_alms : np.ndarray
            Input alm array. It must have shape (n_channels, n_alms, n_components).
        compsep_run : dict
            Component separation parameters including needlet configuration. See 'ilc' for details.
        **kwargs : dict
            Additional keyword arguments for hp.map2alm.

    Returns
    -------
        np.ndarray
            Output maps after needlet domain ILC.
    """
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    if compsep_run["b_squared"]:
        b_ell = b_ell**2

    if compsep_run['save_needlets']:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        os.makedirs(compsep_run["path_out"], exist_ok=True)
        np.save(os.path.join(compsep_run["path_out"], "needlet_bands"), b_ell)

    output_alms = np.zeros((input_alms.shape[1], input_alms.shape[-1]), dtype=complex)
    for j in range(b_ell.shape[0]):
        output_alms += _ilc_needlet_j(config, input_alms, compsep_run, b_ell[j], j, **kwargs)
    
    output_maps = np.array(
        [hp.alm2map(np.ascontiguousarray(output_alms[:, c]), config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
            for c in range(input_alms.shape[-1])]).T
    
    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.,:] = 0.

    return output_maps
        
def _ilc_needlet_j(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: Dict,
    b_ell: np.ndarray,
    nl_scale: int,
    **kwargs: Any
) -> np.ndarray:
    """
    Compute the needlet-scale ILC component separated maps.

    Parameters
    ----------
        config : Configs
            Configuration object. See 'ilc' for details.
        input_alms : np.ndarray
            Input alm array. Shape is (n_channels, n_alms, n_components).
        compsep_run : dict
            Component separation configuration. See 'ilc' for details.
        b_ell : np.ndarray
            Needlet window function. Shape is (lmax+1,).
        nl_scale : int
            Current needlet scale index.
        **kwargs : dict
            Additional keyword arguments.

    Returns
    -------
        np.ndarray
            Output alm array at the requested needlet scale.
    """
    # Determine nside and lmax for this scale
    if "mask" in compsep_run:
        nside_, lmax_ = config.nside, config.lmax
    else:
        if compsep_run["adapt_nside"]:
            nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
        else:
            nside_, lmax_ = config.nside, config.lmax
    
    # Get frequency channels to be adopted in component separation at this scale
    good_channels_nl = _get_good_channels_nl(config, b_ell)

    input_maps_nl = np.zeros((good_channels_nl.shape[0], 12 * nside_**2, input_alms.shape[-1]))
    for n, channel in enumerate(good_channels_nl):
        input_alms_j = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        input_maps_nl[n] = np.array(
            [hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False)
                for c in range(input_alms.shape[-1])]).T

    # Run either MC-ILC or ILC
    if (compsep_run["method"]=="mcilc") or ((compsep_run["method"]=="mc_ilc" or compsep_run["method"]=="mc_cilc") and nl_scale in compsep_run["special_nls"]):
        tracer_nl = get_scalar_tracer_nl(compsep_run["tracers"], nside_, lmax_, b_ell)
        output_maps_nl = _mcilc_maps(config, input_maps_nl, tracer_nl, compsep_run, b_ell, nl_scale=nl_scale)
    else:
        output_maps_nl = _ilc_maps(config, input_maps_nl, compsep_run, b_ell, nl_scale=nl_scale)

    del input_maps_nl

    output_alms_nl = np.array(
        [hp.map2alm(output_maps_nl[:, c], lmax=lmax_, pol=False, **kwargs) for c in range(output_maps_nl.shape[-1])]
    ).T

    if compsep_run["b_squared"]:
        output_alms_j = _needlet_filtering(output_alms_nl, np.ones(lmax_+1), config.lmax)
    else:
        output_alms_j = _needlet_filtering(output_alms_nl, b_ell[:lmax_+1], config.lmax)
        
    return output_alms_j

def _ilc_pixel(config: Configs, input_alms: np.ndarray, compsep_run: Dict, **kwargs) -> np.ndarray:
    """
    Computes ILC maps from input alms in pixel space using the specified component separation method.

    Parameters
    ----------
        config : Configs 
            Configuration parameters. See 'ilc' for details.
        input_alms : np.ndarray
            Input spherical harmonic coefficients of shape (n_channels, n_alm, n_components).
        compsep_run : dict
            Dictionary specifying the component separation configuration. See 'ilc' for details.
        **kwargs : dict
            Additional keyword arguments for hp.map2alm.

    Returns
    -------
        np.ndarray
            Reconstructed sky maps in pixel space.
    """
    npix = 12 * config.nside ** 2
    _, _, n_comps = input_alms.shape
    
    good_channels = _get_good_channels_nl(config, np.ones(config.lmax + 1))
    input_maps = np.zeros((good_channels.shape[0], npix, n_comps))
    
    for n, channel in enumerate(good_channels):
        input_maps[n] = np.array([
            hp.alm2map(np.ascontiguousarray(input_alms[channel, :, c]), config.nside,
                       lmax=config.lmax, pol=False) for c in range(n_comps)
        ]).T


    if compsep_run["method"]=="mcilc":
        tracer = get_scalar_tracer(compsep_run["tracers"])
        output_maps = _mcilc_maps(config, input_maps, tracer, compsep_run, np.ones(config.lmax+1))    
    else:
        output_maps = _ilc_maps(config, input_maps, compsep_run, np.ones(config.lmax+1))
    
    if config.pixel_window_out:
        for c in range(output_maps.shape[1]):
            alm_out = hp.map2alm(output_maps[:,c],lmax=config.lmax, pol=False, **kwargs)
            output_maps[:, c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax,
                                           pol=False, pixwin=True)

    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.,:] = 0.

    return output_maps

def _ilc_maps(config: Configs, input_maps: np.ndarray, compsep_run: Dict,
              b_ell: np.ndarray, nl_scale: Optional[Union[int, None]] = None) -> np.ndarray:
    """
    Computes ILC weights and outputs the cleaned map using ILC methods with inclusion of moments deprojection (if needed).

    Parameters
    ----------
        config : Configs
            Configuration object containing global settings. See 'ilc' for details.
        input_maps : np.ndarray
            Input maps of shape (n_channels, n_pixels, n_components).
        compsep_run : dict
            Dictionary with ILC run configuration. See 'ilc' for details.
        b_ell : np.ndarray
            Needlet harmonic window function. Used if `compsep_run["domain"]` is "needlet".
            If `compsep_run["domain"]` is "pixel", it should be an array of ones with shape (lmax+1,).
        nl_scale : int, optional
            Needlet scale index corresponding to the current ILC run. Used for:
            - Deprojection of moments in cILC.
            - Save weights with proper label.

    Returns
    -------
        np.ndarray
            Cleaned output sky maps of shape (n_pixels, n_components).
    """

    good_channels = _get_good_channels_nl(config, b_ell)
    freqs = np.array(config.instrument.frequency)[good_channels]

    bandwidths = _get_bandwidths(config, good_channels)

    A_cmb = _get_CMB_SED(freqs, units=config.units, bandwidths=bandwidths)
        
    if compsep_run["method"] == "cilc":
        if nl_scale is None:
            compsep_run["A"] = _get_moments_SED(freqs, compsep_run["constraints"]["moments"], beta_d=compsep_run["constraints"]["beta_d"], T_d=compsep_run["constraints"]["T_d"], beta_s=compsep_run["constraints"]["beta_s"], units=config.units, bandwidths=bandwidths)
            compsep_run["e"] = np.array(compsep_run["constraints"]["deprojection"])
        else:
            compsep_run["A"] = _get_moments_SED(freqs, compsep_run["constraints"]["moments"][nl_scale], beta_d=compsep_run["constraints"]["beta_d"][nl_scale], T_d=compsep_run["constraints"]["T_d"][nl_scale], beta_s=compsep_run["constraints"]["beta_s"][nl_scale], units=config.units, bandwidths=bandwidths)
            compsep_run["e"] = np.array(compsep_run["constraints"]["deprojection"][nl_scale])

    elif (compsep_run["method"] == "c_ilc") and (nl_scale is not None) and (nl_scale in compsep_run["special_nls"]):
        compsep_run["A"] = _get_moments_SED(freqs, compsep_run["constraints"]["moments"][compsep_run["special_nls"] == nl_scale], beta_d=compsep_run["constraints"]["beta_d"][compsep_run["special_nls"] == nl_scale], T_d=compsep_run["constraints"]["T_d"][compsep_run["special_nls"] == nl_scale], beta_s=compsep_run["constraints"]["beta_s"][compsep_run["special_nls"] == nl_scale], units=config.units, bandwidths=bandwidths)
        compsep_run["e"] = np.array(compsep_run["constraints"]["deprojection"][compsep_run["special_nls"] == nl_scale])

    elif (compsep_run["method"] == "mc_cilc") and (nl_scale is not None) and (nl_scale not in compsep_run["special_nls"]):
        nl_scale_cilc = nl_scale-len(np.array(compsep_run["special_nls"])[np.array(compsep_run["special_nls"]) < 2])
        compsep_run["A"] = _get_moments_SED(freqs, compsep_run["constraints"]["moments"][nl_scale_cilc], beta_d=compsep_run["constraints"]["beta_d"][nl_scale_cilc], T_d=compsep_run["constraints"]["T_d"][nl_scale_cilc], beta_s=compsep_run["constraints"]["beta_s"][nl_scale_cilc], units=config.units, bandwidths=bandwidths)
        compsep_run["e"] = np.array(compsep_run["constraints"]["deprojection"][nl_scale_cilc])

    cov = get_ilc_cov(input_maps[...,0], config.lmax, compsep_run, b_ell)
    noise_debias = compsep_run["cov_noise_debias"] if compsep_run["domain"] == "pixel" else compsep_run["cov_noise_debias"][nl_scale]
    if noise_debias != 0.:
        cov_n = get_ilc_cov(input_maps[...,compsep_run["noise_idx"]], config.lmax, compsep_run, b_ell)
        cov = cov - noise_debias * cov_n
        del cov_n

    inv_cov = get_inv_cov(cov)
    del cov

    w_ilc = get_ilc_weights(A_cmb, inv_cov, input_maps.shape, compsep_run)
    del inv_cov

    if compsep_run["save_weights"]:
#        if 'path_out' not in compsep_run:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        save_ilc_weights(config, w_ilc, compsep_run,
                         hp.npix2nside(input_maps.shape[-2]), nl_scale=nl_scale)
    
    compsep_run.pop("A", None)
    compsep_run.pop("e", None)

    if compsep_run["ilc_bias"] == 0.:
        output_maps = np.einsum('i,ijk->jk', w_ilc, input_maps)
    else:
        output_maps = np.einsum('ij,ijk->jk', w_ilc, input_maps)

    return output_maps

def _mcilc_maps(config: Configs, input_maps: np.ndarray, tracer: np.ndarray,
                compsep_run: Dict, b_ell: np.ndarray,
                nl_scale: Optional[Union[int, None]] = None) -> np.ndarray:
    """
    Computes foreground-cleaned sky maps using the MC-ILC method.

    Depending on the specified `mc_type`, it delegates to either the CEA-based or 
    random partitioning (RP) implementation.

    Parameters
    ----------
        config : Configs
            Configuration object containing instrument, resolution, and output settings. See 'ilc' for details.
        input_maps : np.ndarray
            Input sky maps of shape (n_channels, n_pixels, n_components).
        tracer : np.ndarray
            Tracer map used to define spatial partitions for independent component separation. 
            It should have shape (n_pixels,).
        compsep_run : dict
            Dictionary specifying the MCILC method and associated parameters:
            - "mc_type": Type of MCILC ('cea' or 'rp').
            - "n_patches": Number of spatial patches.
            - Optional flags such as "save_weights".
            See 'ilc' for details.
        b_ell : np.ndarray
            Needlet harmonic window function. Shape is (lmax+1,).
            Used if `compsep_run["domain"]` is "needlet".
            If `compsep_run["domain"]` is "pixel", it should be an array of ones with shape (lmax+1,).
        nl_scale : int, optional
            Needlet scale index for the current ILC run. Used for:
            - saving weights with proper label.

    Returns
    -------
        np.ndarray
            Cleaned output sky maps of shape (n_pixels, n_components).
    """
    good_channels = _get_good_channels_nl(config, b_ell)
    bandwidths = _get_bandwidths(config, good_channels)

    # Get CMB SED for the good channels
    A_cmb = _get_CMB_SED(np.array(config.instrument.frequency)[good_channels], units=config.units, bandwidths=bandwidths)

    if "cea" in compsep_run["mc_type"]:
        return _mcilc_cea_(config, input_maps, tracer, compsep_run, A_cmb, nl_scale=nl_scale)
    elif "rp" in compsep_run["mc_type"]:
        return _mcilc_rp_(config, input_maps, tracer, compsep_run, A_cmb, nl_scale=nl_scale)

def _mcilc_cea_(config: Configs, input_maps: np.ndarray, tracer: np.ndarray,
                compsep_run: Dict, A_cmb: np.ndarray,
                nl_scale: Optional[Union[int, None]] = None) -> np.ndarray:
    """
    Performs MC-ILC CMB reconstruction using CEA (Cluster of Equal Area) partitioning.

    The sky is divided into non-overlapping spatial patches based on a tracer map and
    the Healpix grid, allowing region-specific ILC weight estimation.

    Parameters
    ----------
        config : Configs
            Configuration object with all global settings. See 'ilc' for details.
        input_maps : np.ndarray
            Input sky maps with shape (n_channels, n_pixels, n_components).
        tracer : np.ndarray
            Tracer map that defines sky partition.
        compsep_run : dict
            Component separation configuration, including:
            - "n_patches": Number of CEA patches.
            - "save_weights": Flag to save weights.
            See 'ilc' for details.
        A_cmb : np.ndarray
            Spectral energy distribution (SED) vector for the CMB.
        nl_scale : int, optional
            Needlet scale index for the current ILC run. Used for:
            - saving weights with proper label.

    Returns
    -------
        np.ndarray
            Output sky map cleaned via region-specific ILC weights, shape (n_pixels, n_components).
    """

    mask_mcilc = compsep_run.get("mask", np.ones(input_maps.shape[-2]))

    patches = _cea_partition(tracer, compsep_run["n_patches"], mask=mask_mcilc)
    if compsep_run["save_patches"] and (compsep_run['nsim'] is None or int(compsep_run['nsim']) == config.nsim_start):
        #if 'path_out' not in compsep_run:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        save_patches(config, patches, compsep_run, nl_scale=nl_scale)

    w_mcilc = get_mcilc_weights(input_maps[...,0], patches, A_cmb, compsep_run)

    if compsep_run["save_weights"]:
        #if 'path_out' not in compsep_run:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        save_ilc_weights(config, w_mcilc, compsep_run, hp.npix2nside(input_maps.shape[-2]), nl_scale=nl_scale)
    
    compsep_run.pop("A", None)
    compsep_run.pop("e", None)

    return np.einsum('ij,ijk->jk', w_mcilc, input_maps)
    
def _mcilc_rp_(config: Configs, input_maps: np.ndarray, tracer: np.ndarray,
               compsep_run: Dict, A_cmb: np.ndarray,
               iterations: int = 30, nl_scale: Optional[Union[int, None]] = None) -> np.ndarray:
    """
    Performs MCILC reconstruction using Random Partitioning (RP) of the sky.

    The tracer map is used to generate multiple random spatial partitions. ILC weights
    are estimated independently for each realization and averaged across iterations
    to reduce bias and improve robustness.

    Parameters
    ----------
        config : Configs
            Configuration object including instrument and output settings. See 'ilc' for details.
        input_maps : np.ndarray
            Input sky maps with shape (n_channels, n_pixels, n_components).
        tracer : np.ndarray
            Tracer map used to guide random partitions.
        compsep_run : dict
            Dictionary with MCILC-specific parameters:
            - "n_patches": Number of partitions per iteration.
            - "save_weights": Whether to save averaged ILC weights.
            See 'ilc' for details.
        A_cmb : np.ndarray
            Spectral energy distribution (SED) of the CMB.
        iterations : int, default=30
            Number of random partition iterations for averaging.
        nl_scale : int or str, optional
            Needlet scale index for the current ILC run. Used for:
            - saving weights with proper label.

    Returns
    -------
        np.ndarray
            Cleaned sky map averaged over random partition realizations,
            shape (n_pixels, n_components).
    """
    output_maps = np.zeros((input_maps.shape[1], input_maps.shape[-1]))
    mask_mcilc = compsep_run.get("mask", np.ones(input_maps.shape[-2]))

    do_save_patches = compsep_run["save_patches"] and (compsep_run['nsim'] is None or int(compsep_run['nsim']) == config.nsim_start)

    if do_save_patches:
        patches_set = []

    for it in range(iterations):  
        patches = _rp_partition(tracer, compsep_run["n_patches"], mask=mask_mcilc)
        if do_save_patches:
            patches_set.append(patches)

        w_mcilc = get_mcilc_weights(input_maps[...,0], patches, A_cmb, compsep_run)
        if compsep_run["save_weights"]:
            if it == 0:
                w_mcilc_save = np.copy(w_mcilc) / iterations
            else:
                w_mcilc_save += w_mcilc / iterations
        output_maps += (np.einsum('ij,ijk->jk', w_mcilc, input_maps) / iterations)
        
    if do_save_patches:
        #if 'path_out' not in compsep_run:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        save_patches(config, np.array(patches_set), compsep_run, nl_scale=nl_scale)

    if compsep_run["save_weights"]:
        #if 'path_out' not in compsep_run:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        save_ilc_weights(config, w_mcilc_save, compsep_run,
                         hp.npix2nside(input_maps.shape[-2]), nl_scale=nl_scale)

    compsep_run.pop("A", None)
    compsep_run.pop("e", None)
    
    return output_maps

def get_ilc_weights(
    A_cmb: np.ndarray,
    inv_cov: np.ndarray,
    input_shapes: Tuple[int, ...],
    compsep_run: Dict
) -> np.ndarray:
    """
    Compute Internal Linear Combination (ILC) weights for a given spectral component (typically CMB).

    Parameters
    ----------
        A_cmb : np.ndarray
            Spectral response vector of the target component (e.g., CMB), shape (n_channels,).
        inv_cov : np.ndarray
            Inverse of the covariance matrix. Can be 2D or 3D depending on bias settings.
        input_shapes : tuple
            Shape of the input map, used to reshape or broadcast weights.
        compsep_run : dict
            Dictionary specifying component separation settings, including:
            - 'A': Optional constraint matrix for deprojection of additional components. Shape (n_components, n_channels).
            - 'e': Deprojection vector. Shape (n_components,).
            - 'ilc_bias': Flag indicating if covariance is pixel-independent (0.0) or not.
            - 'mask': Optional binary mask used for excluding unobserved regions.

    Returns
    -------
        np.ndarray
            ILC weights with shape (n_channels, n_pixels) if ilc_bias is not 0.0,
            otherwise (n_channels,).
    """
    if "A" in compsep_run:
        compsep_run["A"] = np.vstack((A_cmb, compsep_run["A"]))
        compsep_run["e"] = np.insert(compsep_run["e"], 0, 1.)
        if compsep_run["ilc_bias"] == 0.:
            inv_ACA = np.linalg.inv(
                np.einsum("zi,il->zl", compsep_run["A"], np.einsum("ij,lj->il", inv_cov, compsep_run["A"]))
            )
            w_ilc = np.einsum(
                "l,lj->j", compsep_run["e"],
                np.einsum("lz,zj->lj", inv_ACA, np.einsum("zi,ij->zj", compsep_run["A"], inv_cov))
            )
            del inv_ACA
        else:
            inv_ACA = np.linalg.inv(
                np.einsum("zi,ilk->zlk", compsep_run["A"], np.einsum("ijk,lj->ilk", inv_cov, compsep_run["A"])).T
            ).T
            w_ilc=np.zeros((input_shapes[0],input_shapes[-2]))
            w_ = np.einsum(
                "l,ljk->jk", compsep_run["e"],
                np.einsum("lzk,zjk->ljk", inv_ACA, np.einsum("zi,ijk->zjk", compsep_run["A"], inv_cov))
            )
            for i in range(input_shapes[0]):
                if "mask" in compsep_run:
                    w_ilc[i, compsep_run["mask"] > 0.] = w_[i]
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
                    w_ilc[i, compsep_run["mask"] > 0.] = AT_invC[i]/AT_invC_A
                else:
                    w_ilc[i]=hp.ud_grade(AT_invC[i]/AT_invC_A,hp.npix2nside(input_shapes[-2]))
    return w_ilc

def get_mcilc_weights(
    inputs: np.ndarray,
    patches: np.ndarray,
    A_cmb: np.ndarray,
    compsep_run: Dict
) -> np.ndarray:
    """
    Compute weights for the MC-ILC method over spatial patches.

    Parameters
    ----------
        inputs : np.ndarray
            Input sky maps, shape (n_channels, n_pixels).
        patches : np.ndarray
            Integer labels assigning each pixel to a spatial patch.
        A_cmb : np.ndarray
            Spectral response of the CMB component, shape (n_channels,).
        compsep_run : dict
            Dictionary with MCILC options, including optional constraints and bias reduction. See 'ilc' for details.

    Returns
    -------
        np.ndarray
            MCILC weights for each pixel, shape (n_channels, n_pixels).
    """
    mask_mcilc = compsep_run.get("mask", np.ones(inputs.shape[1]))

    cov = get_mcilc_cov(inputs, patches, mask_mcilc, reduce_bias=compsep_run["reduce_mcilc_bias"])

    inv_cov = get_inv_cov(cov)
    del cov

    w_mcilc=np.zeros((inputs.shape[0],inputs.shape[1]))

    if "A" in compsep_run:
        compsep_run["A"] = np.vstack((A_cmb, compsep_run["A"]))
        compsep_run["e"] = np.insert(compsep_run["e"], 0, 1.)
        inv_ACA = np.linalg.inv(
            np.einsum("zi,ilk->zlk", compsep_run["A"], np.einsum("ijk,lj->ilk", inv_cov, compsep_run["A"])).T
        ).T
        w_mcilc[:, mask_mcilc > 0.0] = np.einsum(
            "l,ljk->jk", compsep_run["e"],
            np.einsum("lzk,zjk->ljk", inv_ACA, np.einsum("zi,ijk->zjk", compsep_run["A"], inv_cov))
        )
        del inv_ACA
    else:
        AT_invC = np.einsum('j,ijk->ik', A_cmb, inv_cov) # np.sum(inv_cov,axis=1)
        AT_invC_A = np.einsum('j,ijk, i->k', A_cmb, inv_cov, A_cmb) #np.sum(inv_cov,axis=(0,1))
        for i in range(inputs.shape[0]):
            w_mcilc[i, mask_mcilc > 0.] = AT_invC[i]/AT_invC_A
        del AT_invC, AT_invC_A
    
    del inv_cov
    return w_mcilc


def get_ilc_cov(
    input_maps: np.ndarray,
    lmax: int,
    compsep_run: Dict,
    b_ell: np.ndarray
) -> np.ndarray:
    """
    Compute the input covariance matrix for ILC.

    Parameters
    ----------
        input_maps : np.ndarray
            Input sky maps, shape (n_channels, n_pixels).
        lmax : int
            Maximum multipole.
        compsep_run : dict
            Dictionary controlling ILC behavior (bias flags, mask, etc.). See 'ilc' for details.
        b_ell : np.ndarray
            Needlet harmonic window function, if applicable.

    Returns
    -------
        np.ndarray
            Covariance matrix, either global (2D) or pixel-dependent (3D).
    """
    if compsep_run["ilc_bias"] == 0.:
        if "mask" in compsep_run:
            cov=np.mean(np.einsum('ik,jk->ijk', (input_maps * compsep_run["mask"])[:, compsep_run["mask"] > 0.], (input_maps * compsep_run["mask"])[:, compsep_run["mask"] > 0.]),axis=-1)
        else:
            cov=np.mean(np.einsum('ik,jk->ijk', input_maps, input_maps),axis=-1)
    else:
        cov = _get_local_cov(
            input_maps, lmax, compsep_run["ilc_bias"], b_ell=b_ell,
            mask=compsep_run.get("mask"), reduce_bias=compsep_run["reduce_ilc_bias"]
        )
        if "mask" in compsep_run and cov.shape[-1] == input_maps.shape[-1]:
                cov = np.copy(cov[...,compsep_run["mask"] > 0.])
    return cov

def get_mcilc_cov(
    inputs: np.ndarray,
    patches: np.ndarray,
    mask_mcilc: np.ndarray,
    reduce_bias: bool = True,
    mcilc_rings: int = 4
) -> np.ndarray:
    """
    Compute pixel-wise covariance matrices for MC-ILC based on patching and optional donut-masking.

    Parameters
    ----------
        inputs : np.ndarray
            Input sky maps, shape (n_channels, n_pixels).
        patches : np.ndarray
            Pixel-wise patch labels.
        mask_mcilc : np.ndarray
            Binary mask to include/exclude pixels.
        reduce_bias : bool, default=True
            Whether to apply donut masking to reduce MCILC bias.
        mcilc_rings : int, default=4
            Number of HEALPix neighbor rings to exclude when reducing bias.

    Returns
    -------
        np.ndarray
            Covariance matrix per pixel, shape (n_channels, n_channels, n_valid_pixels).
    """
    cov=np.zeros((inputs.shape[0], inputs.shape[0], inputs.shape[-1]))
    
    if reduce_bias:
        neigh = hp.get_all_neighbours(hp.npix2nside(inputs.shape[1]), np.argwhere(mask_mcilc > 0.)[:, 0])
        for pix_ in np.argwhere(mask_mcilc > 0.)[:, 0]:
            donut = np.ones(inputs.shape[1])
            donut[pix_] = 0.   
            if mcilc_rings > 0:
                donut[neigh[:,pix_]]=0.
            if mcilc_rings > 1:
                count=1
                neigh_=neigh[:,pix_]
                while count < mcilc_rings:
                    neigh_=neigh[:,neigh_].flatten()
                    donut[neigh_]=0.
                    count=count+1
            patch_mask = (patches==patches[pix_]) & (donut>0.) & (mask_mcilc>0.)
            cov[...,pix_] = np.cov((inputs * mask_mcilc)[:,patch_mask],rowvar=True)
    else:
        for patch in np.unique(patches):
            patch_mask = (patches == patch) & (mask_mcilc > 0.)
            patch_cov = np.mean(np.einsum('ik,jk->ijk', (inputs * mask_mcilc)[:,patch_mask], (inputs * mask_mcilc)[:,patch_mask]),axis=2)
            #cov[...,(patches==patch)] = np.repeat(patch_cov[:, :, np.newaxis], np.sum(patches==patch), axis=2)
            cov[...,patch_mask] = np.repeat(patch_cov[:, :, np.newaxis], np.sum(patch_mask), axis=2)

    return cov[...,mask_mcilc > 0.]

def get_inv_cov(cov: np.ndarray) -> np.ndarray:
    """
    Invert covariance matrices for ILC or MC-ILC processing.

    Parameters
    ----------
        cov : np.ndarray
            Covariance matrix of shape (n_channels, n_channels) or (n_channels, n_channels, n_pixels).

    Returns
    -------
        np.ndarray
            Inverted covariance matrix with same shape.
    """
    if cov.ndim == 2:
        inv_cov=np.linalg.inv(cov)
    elif cov.ndim == 3:
        inv_cov=np.linalg.inv(cov.T).T
    return inv_cov


__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]
                    

