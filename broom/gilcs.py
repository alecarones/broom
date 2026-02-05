import numpy as np
import healpy as hp
import sys
from .configurations import Configs
from .routines import _EB_to_QU, _E_to_QU, _B_to_QU, obj_to_array, array_to_obj, _log, _get_bandwidths
from .saving import _save_compsep_products, _get_full_path_out, save_ilc_weights, _get_full_path_nuiscov, update_and_save_nuiscov_serial, load_nuiscov
from .needlets import _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering, _get_good_channels_nl
from .masking import _downgrade_mask
from .ilcs import get_ilc_cov
from .seds import _get_CMB_SED
import scipy
from numpy import linalg as lg
from types import SimpleNamespace
import os
from typing import Any, Dict, Optional, Union, List, Tuple

def gilc(config: Configs, input_alms: SimpleNamespace, compsep_run: Dict[str, Any], **kwargs) -> Optional[SimpleNamespace]:
    """
    Performs Generalized Internal Linear Combination (GILC) component separation on scalar fields.

    Parameters
    ----------
        config: Configs
            Configuration object with general settings. It includes:
                - lmax : int, maximum multipole for the component separation.
                - nside : int, HEALPix resolution parameter for compsep products.
                - fwhm_out : float, full width at half maximum of the output beam in arcminutes.
                - pixel_window_out : bool, whether to apply a pixel window to the output maps.
                - field_out : str, desired output fields (e.g., "T", "E", "B", "QU", "TQU", "QU_E", "QU_B").
                - save_compsep_products : bool, whether to save component separation products.
                - return_compsep_products : bool, whether to return component separation products.
                - path_outputs : str, path to save the output files.
            
        input_alms: SimpleNamespace
            Input multifrequency alms of the scalar fields. 
            Each attribute should be a numpy array of shape (n_channels, (n_fields), n_alms, n_components), of the noise maps. 
            with n_fields > 1 if multiple scalars are provided.
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
            - depro_cmb : Optional[Union[float, list, np.ndarray]], deprojection factor for CMB (scalar or list per needlet bands). 
                    Default is None.
            - m_bias : Optional[Union[float, list, np.ndarray]], if not zero, it will include m_bias more (if m_bias > 0) 
                    or less (if m_bias < 0) modes in the reconstructed GNILC maps. Default is 0.
                    It can be a list if different values are needed for different needlet bands.
            - nuisance : Union[str, List[str]], name or list of names of the nuisance components to be used for the nuisance covariance.
            - needlet_config: Dictionary containing needlet settings. Needed if domain is "needlet". It should include:
                - "needlet_windows": Type of needlet windows ('cosine', 'standard', 'mexican').
                - "ell_peaks": List of integers defining multipoles of peaks for needlet bands (required for 'cosine').
                - "width": Width of the needlet windows (required for 'standard' and 'mexican').
                - "merging_needlets": Integer or list of integers defining ranges of needlets to be merged.
            - b_squared : bool, whether to square the needlet windows. Default is False.            
            - adapt_nside : bool, whether to adapt the nside based on the needlet windows. Default is False.
            - mask : Optional[np.ndarray], mask to apply to the maps (if available).
            - cov_noise_debias : Optional[Union[float, list, np.ndarray]],
                If not zero, it will debias the covariances by subtracting a term cov_noise_debias * noise_covariance.
                It can be a scalar or a list per needlet bands. Default is 0.
            
        **kwargs: 
            Dictionary of additional keyword arguments to pass to healpy function 'map2alm'.

    Returns
    -------
        Optional[SimpleNamespace]: 
            Output map object with reconstructed foreground maps if config.return_compsep_products is True.
    """
    if not hasattr(input_alms, "total"):
        raise ValueError("input_alms must have 'total' attribute containing the total input alms for GILC.")

    input_attrs = obj_to_array(input_alms, return_attributes=True)

    compsep_run = _standardize_gnilc_run(compsep_run, input_alms.total.shape[0], config.lmax)
    
    if not ("load_nuisance_covariance" in compsep_run and compsep_run["load_nuisance_covariance"]):
        compsep_run["nuis_idx"] = get_nuisance_idx(input_attrs, compsep_run, config.verbose)
    if np.any(np.array(compsep_run["cov_noise_debias"]) != 0.):
        if not ("load_noise_covariance" in compsep_run and compsep_run["load_noise_covariance"]):
            if not hasattr(input_alms, "noise"):
                raise ValueError("The input_alms object must have 'noise'' attribute for debiasing the covariance.")
            compsep_run["noise_idx"] = input_attrs.index('noise')
#    print("Using nuisance alms with indices:", compsep_run["nuis_idx"])

    output_maps = _gilc(config, obj_to_array(input_alms), compsep_run, **kwargs)
    
    output_maps = _gilc_post_processing(config, output_maps, compsep_run, **kwargs)

    compsep_run.pop("nuis_idx", None)
    compsep_run.pop("noise_idx", None)

    outputs = array_to_obj(output_maps, input_attrs)
    del output_maps

    return outputs, compsep_run

def fgd_diagnostic(config: Configs, input_alms: SimpleNamespace, compsep_run: Dict[str, Any], **kwargs) -> Optional[SimpleNamespace]:
    """
    Return diagnostic maps of the foreground complexity for the provided scalar fields.

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
            Input multifrequency alms of the scalar fields. 
            Each attribute should be a numpy array of shape (n_channels, (n_fields), n_alms, n_components), 
            with n_fields > 1 if multiple scalars are provided.
            It should contain:
            - total : np.ndarray, alms of the total input maps.
            - nuisance : Optional[np.ndarray], alms of the nuisance maps (if available).
            Alternatively to nuisance, it can contain:
            - cmb : np.ndarray, alms of the CMB maps.
            - noise : np.ndarray, alms of the noise maps.
    
        compsep_run: Dict[str, Any]
            Dictionary with diagnostic parameters. It should include:
                - domain : str, either "pixel" or "needlet" for the component separation domain.
                - nuisance : Union[str, List[str]], name or list of names of the nuisance components to be used for the nuisance covariance.
                            Default is ["noise", "cmb"].
                - needlet_config: Dictionary containing needlet settings. Needed if domain is "needlet". It should include:
                    - "needlet_windows": Type of needlet windows ('cosine', 'standard', 'mexican').
                    - "ell_peaks": List of integers defining multipoles of peaks for needlet bands (required for 'cosine').
                    - "width": Width of the needlet windows (required for 'standard' and 'mexican').
                    - "merging_needlets": Integer or list of integers defining ranges of needlets to be merged.
                - b_squared : bool, whether to square the needlet windows. Default is False.            
                - adapt_nside : bool, whether to adapt the nside based on the needlet windows. Default is False.
                - mask : Optional[np.ndarray], mask to apply to the maps (if available).
            
        **kwargs: 
            Dictionary of additional keyword arguments to pass to healpy function 'map2alm'.

    Returns
    -------
        Optional[SimpleNamespace]
            Object containing foreground diagnostic maps.
    """
    if not hasattr(input_alms, "total"):
        raise ValueError("input_alms must have 'total' attribute containing the total input alms for fgd_diagnostic.")

    if not "nuisance" in compsep_run:
        compsep_run["nuisance"] = ["noise", "cmb"]  

    input_attrs = obj_to_array(input_alms, return_attributes=True)        
     
    if not ("load_nuisance_covariance" in compsep_run and compsep_run["load_nuisance_covariance"]):
        compsep_run["nuis_idx"] = get_nuisance_idx(input_attrs, compsep_run, config.verbose)
        if isinstance(compsep_run["nuis_idx"], int):
            nuis_alms = (obj_to_array(input_alms))[...,compsep_run["nuis_idx"]]
        elif isinstance(compsep_run["nuis_idx"], list):
            nuis_alms = np.copy(obj_to_array(input_alms))[...,compsep_run["nuis_idx"][0]]
            if len(compsep_run["nuis_idx"]) > 1:
                for idx in compsep_run["nuis_idx"][1:]:
                    nuis_alms += (obj_to_array(input_alms))[...,idx]
        compsep_run["nuis_idx"] = 1

    if np.any(np.array(compsep_run["cov_noise_debias"]) != 0.):
        if not ("load_noise_covariance" in compsep_run and compsep_run["load_noise_covariance"]):
            if not hasattr(input_alms, "noise"):
                raise ValueError("The input_alms object must have 'noise'' attribute for debiasing the covariance.")
            compsep_run["noise_idx"] = input_attrs.index('noise')
            noi_alms = (obj_to_array(input_alms))[...,compsep_run["noise_idx"]]
            if not ("load_nuisance_covariance" in compsep_run and compsep_run["load_nuisance_covariance"]):
                compsep_run["noise_idx"] = 2
            else:
                compsep_run["noise_idx"] = 1

    input_alms_for_diagn = input_alms.total[...,np.newaxis]
    if not ("load_nuisance_covariance" in compsep_run and compsep_run["load_nuisance_covariance"]):
        input_alms_for_diagn = np.concatenate([input_alms_for_diagn, nuis_alms[...,np.newaxis]], axis=-1)
        del nuis_alms
    if np.any(np.array(compsep_run["cov_noise_debias"]) != 0.):
        if not ("load_noise_covariance" in compsep_run and compsep_run["load_noise_covariance"]):
            input_alms_for_diagn = np.concatenate([input_alms_for_diagn, noi_alms[...,np.newaxis]], axis=-1)
            del noi_alms

    output_maps = _fgd_diagnostic(config, input_alms_for_diagn, compsep_run)
    del input_alms_for_diagn

    compsep_run.pop("nuis_idx", None)
    compsep_run.pop("noise_idx", None)

    outputs = SimpleNamespace(m=output_maps)

    del output_maps

    return outputs, compsep_run

def get_nuisance_idx(
    input_attrs: List[str],
    compsep_run: Dict[str, Any],
    verbose: bool = False,
) -> Union[int, List[int]]:
    """
    Determines the index or indices of nuisance alms based on the input alms object and component separation parameters.

    Parameters
    ----------
        input_attrs: List[str]
            List of attribute names in the input_alms object.
        compsep_run: Dict[str, Any]
            Component separation parameters including 'nuisance'.
        verbose: bool, optional
            If True, prints additional information about the nuisance alms being used. Default is False.
    
    Returns
    -------
        Union[int, List[int]]
            Index or list of indices for nuisance alms.
    
    Raises  
    ------
    ValueError
        If the input_alms object does not have the required attributes for nuisance covariance.
    """
    if not "nuisance" in compsep_run or compsep_run["nuisance"] is None:
        raise ValueError("compsep_run must contain 'nuisance' key with the nuisance components to be used.")

    if isinstance(compsep_run["nuisance"], str) or len(compsep_run["nuisance"]) == 1:
        nuis_comp = compsep_run["nuisance"] if isinstance(compsep_run["nuisance"], str) else compsep_run["nuisance"][0]
        if nuis_comp not in input_attrs:
            raise ValueError(f"The input_alms object does not have '{nuis_comp}' attribute needed for nuisance covariance.")
        nuis_idx = input_attrs.index(nuis_comp)
        if verbose:
            _log(f"GILC/FGD will use '{nuis_comp}' alms for the nuisance covariance.")
    else:
        nuis_idx = []
        for nuis_comp in compsep_run["nuisance"]:
            if nuis_comp not in input_attrs:
                raise ValueError(f"The input_alms object does not have '{nuis_comp}' attribute needed for nuisance covariance.")
            nuis_idx.append(input_attrs.index(nuis_comp))
        if verbose:
            _log(f"GILC/FGD will use {[comp for comp in compsep_run['nuisance']]} alms for the nuisance covariance.")
    
    return nuis_idx

def _gilc_post_processing(config: Configs, output_maps: np.ndarray, compsep_run: Dict[str, Any], **kwargs) -> np.ndarray:
    """
    Applies post-processing steps to GILC output maps, such as converting from E/B to Q/U or masking.

    Parameters
    ----------
        config: Configs
            Configuration object with general settings. See 'gilc' function for details.
        output_maps: np.ndarray
            Raw output maps from GILC. Shape must be (n_channels, (n_fields), npix, n_components).
        compsep_run: Dict[str, Any]
            Component separation parameters. See 'gilc' function for details.
        **kwargs: 
            Additional arguments for EB/QU conversions.

    Returns
    -------
        np.ndarray
            Processed output maps.
    """
    if output_maps.ndim == 4 and (
        (output_maps.shape[1] == 2 and config.field_out == "QU") or
        (output_maps.shape[1] == 3 and config.field_out == "TQU")
    ):
        outputs = np.zeros_like(output_maps)
        for f, c in np.ndindex(output_maps.shape[0],output_maps.shape[-1]):
            outputs[f,...,c] = _EB_to_QU(output_maps[f,...,c],config.lmax, **kwargs)

        if "mask" in compsep_run:
            outputs[:,:,compsep_run["mask"] == 0.,:] = 0.
        return outputs

    elif (output_maps.ndim==3) and (config.field_out in ["QU_E", "QU_B"]):
        output = np.zeros((output_maps.shape[0], 2, output_maps.shape[1], output_maps.shape[-1]))
        for f, c in np.ndindex(output_maps.shape[0],output_maps.shape[-1]):
            if config.field_out == "QU_E":
                output[f,...,c] = _E_to_QU(output_maps[f,:,c],config.lmax, **kwargs)
            elif config.field_out=="QU_B":
                output[f,...,c] = _B_to_QU(output_maps[f,:,c],config.lmax, **kwargs)
        if "mask" in compsep_run:
            output[:,:,compsep_run["mask"] == 0.,:] = 0.
        return output
        
    else:
        if "mask" in compsep_run:
            output_maps[...,compsep_run["mask"] == 0.,:] = 0.
        return output_maps

def _gilc(config: Configs, input_alms: np.ndarray, compsep_run: Dict[str, Any], **kwargs) -> np.ndarray:
    """
    Apply GILC component separation routine to each provided scalar field.

    Parameters
    ----------
        config: Configs
            Configuration object with general settings. See 'gilc' function for details.
        input_alms: np.ndarray
            Input alms (3D or 4D array). 
            Shape should be (n_channels, n_fields, n_alms, n_components) for 4D or (n_channels, n_alms, n_components) for 3D.
        compsep_run: Dict[str, Any]
            Parameters for the GILC component separation run. See 'gilc' function for details.
        **kwargs: 
            Additional keyword arguments for healpy function map2alm.

    Returns
    -------
        np.ndarray
            Output separated component maps. 
            Shape will be (n_channels_out, (n_fields), npix, n_components).
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

    if input_alms.ndim == 4:
        output_maps = np.zeros((len(compsep_run["channels_out"]), input_alms.shape[1], 12 * config.nside**2, input_alms.shape[-1]))
        for i in range(input_alms.shape[1]):
            compsep_run["field"] = fields_ilc[i]
            output_maps[:,i] = _gilc_scalar(config, input_alms[:, i], compsep_run, **kwargs)
    elif input_alms.ndim == 3:
        compsep_run["field"] = fields_ilc[0]
        output_maps = _gilc_scalar(config, input_alms, compsep_run, **kwargs)

    del compsep_run["field"]
    
    return output_maps

def compute_and_update_nuisance_covariance(config: Configs, nuis_alms: np.ndarray, nuis_case: Dict[str, Any], nsim: Optional[Union[int, str]], **kwargs) -> None:
    """
    Computes the nuisance covariance matrix and updates the stored estimate in disk.

    Parameters
    ----------
        config: Configs
            Configuration object with general settings.
        nuis_alms: np.ndarray
            Nuisance alms array of shape (n_channels, (n_fields), n_alms).
        nuis_case: Dict[str, Any]
            Dictionary with nuisance covariance parameters.
        nsim: Optional[Union[int, str]]
            Simulation number identifier for covariance estimation.
        **kwargs:
            Additional keyword arguments for alm/map conversions.

    Returns
    -------
        None
    """

    if nuis_alms.ndim == 3:
        if nuis_alms.shape[1] == 3:
            fields_ilc = ["T", "E", "B"]
        elif nuis_alms.shape[1] == 2:
            fields_ilc = ["E", "B"]
    elif nuis_alms.ndim == 2:
        if config.field_out in ["T", "E", "B"]:
            fields_ilc = [config.field_out]
        elif config.field_out in ["QU_E", "QU_B"]:
            fields_ilc = [config.field_out[-1]]

    if nuis_alms.ndim == 3:
        for i in range(nuis_alms.shape[1]):
            nuis_case["field"] = fields_ilc[i]
            _nuiscov_scalar(config, nuis_alms[:, i], nuis_case, **kwargs)
    elif nuis_alms.ndim == 2:
        nuis_case["field"] = fields_ilc[0]
        _nuiscov_scalar(config, nuis_alms, nuis_case, **kwargs)

    del nuis_case["field"]


def _fgd_diagnostic(config: Configs, input_alms: np.ndarray, compsep_run: Dict[str, Any]) -> np.ndarray:
    """
    Foreground diagnostic map generation.

    Parameters
    ----------
        config: Configs
            Configuration object with general settings. See 'fgd_diagnostic' function for details.
        input_alms: np.ndarray
            Input alms (4D or 3D array). 
            Shape should be (n_channels, (n_fields), n_alms, n_comps) being 4D if multiple fields are provided, 
            or 3D if only one field is provided.
        compsep_run: Dict[str, Any] 
            Parameters for foreground diagnostic. See 'fgd_diagnostic' function for details.

    Returns
    -------
        np.ndarray
            Output diagnostic maps.
            Shape will be ((n_fields), npix) if domain is "pixel", or ((n_fields), n_bands, npix) if domain is "needlet".
    """
    if input_alms.ndim == 4:
        if compsep_run["domain"]=="needlet":
            nls_number = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax).shape[0]
            output_maps = np.zeros((input_alms.shape[1], nls_number, 12 * config.nside**2))
        else:
            output_maps = np.zeros((input_alms.shape[1], 12 * config.nside**2))

        for i in range(input_alms.shape[1]):
            output_maps[i] = _fgd_diagnostic_scalar(config, input_alms[:, i], compsep_run)

    elif input_alms.ndim == 3:
        output_maps = _fgd_diagnostic_scalar(config, input_alms, compsep_run)

    return output_maps

def _standardize_gnilc_run(compsep_run: Dict[str, Any], n_freqs: int, lmax: int) -> Dict[str, Any]:
    """
    Standardizes the `compsep_run` dictionary for GILC/FGD operations.

    Parameters
    ----------
        compsep_run : dict
            Dictionary of component separation configuration parameters. See `gilc` for details.
        n_freqs : int
            Number of frequency channels in the input data.
        lmax : int
            Maximum multipole to consider.

    Returns
    -------
        dict
            Standardized compsep_run dictionary.
    """
    if not "channels_out" in compsep_run:
        compsep_run["channels_out"] = list(range(n_freqs))
    else: 
        if np.any(np.array(compsep_run["channels_out"]) >= n_freqs):
            raise ValueError("Some of the requested channels_out are not present in the input maps.")

    if compsep_run["domain"]=="needlet":
        nls_number = _get_needlet_windows_(compsep_run["needlet_config"], lmax).shape[0]

    # CMB deprojection setup    
    if "depro_cmb" not in compsep_run or compsep_run["depro_cmb"] is None:
        compsep_run["depro_cmb"] = None if compsep_run["domain"] == "pixel" else np.repeat(None, nls_number)
    else:
        if compsep_run["domain"] == "pixel":
            if isinstance(compsep_run["depro_cmb"], (list, np.ndarray)):
                raise ValueError("depro_cmb must be a scalar or None when domain is pixel.")
        elif compsep_run["domain"] == "needlet":
            if isinstance(compsep_run["depro_cmb"], (int, float)):
                compsep_run["depro_cmb"] = np.repeat(compsep_run["depro_cmb"], nls_number)
            elif isinstance(compsep_run["depro_cmb"], list):
                compsep_run["depro_cmb"] = (compsep_run["depro_cmb"] + [None] * nls_number)[:nls_number]
            elif isinstance(compsep_run["depro_cmb"], np.ndarray):
                _len = nls_number - compsep_run["depro_cmb"].shape[0]
                if _len > 0:
                    compsep_run["depro_cmb"] = np.append(compsep_run["depro_cmb"], [None] * _len)
                compsep_run["depro_cmb"] = compsep_run["depro_cmb"][:nls_number]
            else:
                raise ValueError("depro_cmb must be a scalar, a list or a np.ndarray if domain is needlet.")

    # Bias setup
    if "m_bias" not in compsep_run or compsep_run["m_bias"] is None:
        compsep_run["m_bias"] = 0 if compsep_run["domain"] == "pixel" else np.repeat(0, nls_number)
    else:
        if compsep_run["domain"]=="pixel":
            if isinstance(compsep_run["m_bias"], (list, np.ndarray)):
                raise ValueError("m_bias must be a scalar when domain is pixel.")
        elif compsep_run["domain"]=="needlet":
            if isinstance(compsep_run["m_bias"], int):
                compsep_run["m_bias"] = np.repeat(compsep_run["m_bias"], nls_number)
            elif isinstance(compsep_run["m_bias"], list):
                compsep_run["m_bias"] = (compsep_run["m_bias"] + [0] * nls_number)[:nls_number]
            elif isinstance(compsep_run["m_bias"], np.ndarray):
                _len = nls_number - compsep_run["m_bias"].shape[0]
                if _len > 0:
                    compsep_run["m_bias"] = np.append(compsep_run["m_bias"], [0] * _len)
                compsep_run["m_bias"] = compsep_run["m_bias"][:nls_number]    
            else:
                raise ValueError("m_bias must be a scalar, a list or a np.ndarray if domain is needlet.")

    if "nuisance" not in compsep_run:
        compsep_run["nuisance"] = ["noise", "cmb"]           
     
    return compsep_run

def _gilc_scalar(config: Configs, input_alms: np.ndarray, compsep_run: Dict[str, Any], **kwargs) -> np.ndarray:
    """
    Apply GILC component separation in pixel or needlet domain.

    Parameters
    ----------
        config : Configs
            Configuration object. See `gilc` for details.
        input_alms : np.ndarray
            Input alm coefficients. Shape should be (n_channels, n_alms, n_components).
        compsep_run : dict
            Dictionary with GILC parameters. See `gilc` for details.

    Returns
    -------
        np.ndarray
            Separated component maps. Shape will be (n_channels_out, npix, n_components).
    """
    if compsep_run["domain"] == "pixel":
        return _gilc_pixel(config, input_alms, compsep_run, **kwargs)
    elif compsep_run["domain"] == "needlet":
        return _gilc_needlet(config, input_alms, compsep_run, **kwargs)
    else:
        raise ValueError(f"Unsupported domain: {compsep_run['domain']}")

def _nuiscov_scalar(config: Configs, nuis_alms: np.ndarray, compsep_run: Dict[str, Any], **kwargs) -> None:
    """
    Compute and store nuisance covariance in pixel or needlet domain.

    Parameters
    ----------
        config : Configs
            Configuration object. See `compute_and_update_nuisance_covariance` for details.
        nuis_alms : np.ndarray
            Nuisance alm coefficients. Shape should be (n_channels, n_alms).
        compsep_run : dict
            Dictionary with parameters for nuisance covariance computation. See `compute_and_update_nuisance_covariance` for details.

    Returns
    -------
        None
    """
    if compsep_run["domain"] == "pixel":
        _nuiscov_pixel(config, nuis_alms, compsep_run, **kwargs)
    elif compsep_run["domain"] == "needlet":
        _nuiscov_needlet(config, nuis_alms, compsep_run, **kwargs)
    else:
        raise ValueError(f"Unsupported domain: {compsep_run['domain']}")


def _fgd_diagnostic_scalar(config: Configs, input_alms: np.ndarray, compsep_run: Dict[str, Any]) -> np.ndarray:
    """
    Get Foreground Diagnostic maps for provided scalar field either in pixel or needlet domain.

    Parameters
    ----------
        config : Configs
            Configuration object. See `fgd_diagnostic` for details.
        input_alms : np.ndarray
            Input alm coefficients. Shape should be (n_channels, n_alms).
        compsep_run : dict
            Dictionary with run parameters. See `fgd_diagnostic` for details.

    Returns
    -------
        np.ndarray
            Foreground diagnostic maps. Shape will be (npix) if domain is "pixel", or (n_bands, npix) if domain is "needlet".
    """
    if compsep_run["domain"] == "pixel":
        return _fgd_diagnostic_pixel(config, input_alms, compsep_run)
    elif compsep_run["domain"] == "needlet":
        return _fgd_diagnostic_needlet(config, input_alms, compsep_run)
    else:
        raise ValueError(f"Unsupported domain: {compsep_run['domain']}")

def _gilc_pixel(config: Configs, input_alms: np.ndarray, compsep_run: Dict[str, Any], **kwargs) -> np.ndarray:
    """
    GILC in pixel domain.

    Parameters
    ----------
        config : Configs
            Configuration object. See `gilc` for details.
        input_alms : np.ndarray
            Input spherical harmonic coefficients. Shape should be (n_channels, n_alms, n_components).
        compsep_run : dict
            Component separation settings. See `gilc` for details.
        kwargs : dict
            Extra keyword arguments passed to alm/map conversion routines.

    Returns
    -------
        np.ndarray
            Output map with separated components. Shape will be (n_channels_out, npix, n_components).
    """

    compsep_run["good_channels"] = _get_good_channels_nl(config, np.ones(config.lmax+1))

    input_maps = np.zeros((compsep_run["good_channels"].shape[0], 12 * config.nside**2, input_alms.shape[-1]))
    for n, channel in enumerate(compsep_run["good_channels"]):
        input_maps[n] = np.array([hp.alm2map(np.ascontiguousarray(input_alms[channel, :, c]), config.nside, lmax=config.lmax, pol=False) for c in range(input_alms.shape[-1])]).T

    output_maps = _gilc_maps(
        config,
        input_maps,
        compsep_run,
        np.ones(config.lmax + 1),
        depro_cmb=compsep_run["depro_cmb"],
        m_bias=compsep_run["m_bias"],
        noise_debias=compsep_run["cov_noise_debias"],
        mask=compsep_run.get("mask", None)
    )
    
    del compsep_run['good_channels']

    if config.pixel_window_out:
        for f, c in np.ndindex(output_maps.shape[0],output_maps.shape[-1]):
            alm_out = hp.map2alm(output_maps[f,:,c],lmax=config.lmax, pol=False, **kwargs)
            output_maps[f,:,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=False, pixwin=True)

    if "mask" in compsep_run:
        output_maps[:,compsep_run["mask"] == 0.,:] = 0.

    return output_maps

def _nuiscov_pixel(config: Configs, nuis_alms: np.ndarray, compsep_run: Dict[str, Any], **kwargs) -> None:
    """
    Nuisance covariance computation in pixel domain.

    Parameters
    ----------
        config : Configs
            Configuration object. See `compute_and_update_nuisance_covariance' for details.
        nuis_alms : np.ndarray
            Input spherical harmonic coefficients for nuisance. Shape should be (n_channels, n_alms).
        compsep_run : dict
            Settings for nuisance covariance computation. See `compute_and_update_nuisance_covariance` for details.
        kwargs : dict
            Extra keyword arguments passed to alm/map conversion routines.

    """

    compsep_run["good_channels"] = _get_good_channels_nl(config, np.ones(config.lmax+1))

    input_maps = np.zeros((compsep_run["good_channels"].shape[0], 12 * config.nside**2))
    for n, channel in enumerate(compsep_run["good_channels"]):
        input_maps[n] = hp.alm2map(np.ascontiguousarray(nuis_alms[channel, :]), config.nside, lmax=config.lmax, pol=False)

    _nuiscov(
        config,
        input_maps,
        compsep_run,
        np.ones(config.lmax + 1), mask=compsep_run.get("mask", None)
    )
    
    del compsep_run['good_channels']

    return None

def _fgd_diagnostic_pixel(config: Configs, input_alms: np.ndarray, compsep_run: dict) -> np.ndarray:
    """
    Generate pixel-domain diagnostic maps from input alms.

    Parameters
    ----------
        config: Configs
            Configuration object with general settings, including nside and lmax. See `fgd_diagnostic` for details.
        input_alms: np.ndarray
            Input spherical harmonic coefficients. Shape should be (n_channels, n_alms, n_comps).
        compsep_run: dict
            Dictionary with component separation parameters, including mask (if available). See `fgd_diagnostic` for details.


    Returns
    -------
        np.ndarray
            Diagnostic map of foregrounc complexity.
    """

    compsep_run["good_channels"] = _get_good_channels_nl(config, np.ones(config.lmax+1))

    input_maps = np.zeros((compsep_run["good_channels"].shape[0], 12 * config.nside**2, input_alms.shape[-1]))
    for n, channel in enumerate(compsep_run["good_channels"]):
        input_maps[n] = np.array([hp.alm2map(np.ascontiguousarray(input_alms[channel, :, c]), config.nside, lmax=config.lmax, pol=False) for c in range(input_alms.shape[-1])]).T

    output_maps = _get_diagnostic_maps(config, input_maps, compsep_run, np.ones(config.lmax+1), noise_debias=compsep_run["cov_noise_debias"], mask=compsep_run.get("mask", None))

    del compsep_run['good_channels']

    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.] = 0.

    return output_maps

def _gilc_needlet(config: Configs, input_alms: np.ndarray, compsep_run: dict, **kwargs) -> np.ndarray:
    """
    Apply GILC in the needlet domain.

    Parameters
    ----------
        config: Configs
            Configuration object which includes general settings like nside and lmax. See `gilc` for details.
        input_alms: np.ndarray
            Input alms with shape (n_channels, n_alms, n_components).
        compsep_run: dict
            GILC parameters including needlet configuration. See `gilc` for details.
        **kwargs: dict
            Extra parameters for alm2map/map2alm.

    Returns
    ----------
        np.ndarray
            Output GNILC maps with shape (n_channels_out, npix, n_components).
    """
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    if compsep_run["b_squared"]:
        b_ell = b_ell**2

    if compsep_run['save_needlets']:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        os.makedirs(compsep_run["path_out"], exist_ok=True)
        np.save(os.path.join(compsep_run["path_out"], "needlet_bands"), b_ell)
        
    output_alms = np.zeros((len(compsep_run["channels_out"]), input_alms.shape[1], input_alms.shape[-1]), dtype=complex)
    for j in range(b_ell.shape[0]):
        output_alms += _gilc_needlet_j(
            config, input_alms, compsep_run,
            b_ell[j], j, depro_cmb=compsep_run["depro_cmb"][j], m_bias=compsep_run["m_bias"][j], 
            noise_debias=compsep_run["cov_noise_debias"][j], **kwargs
        )

    output_maps = np.zeros((output_alms.shape[0], 12 * config.nside**2, output_alms.shape[-1]))
    for f, c in np.ndindex(output_alms.shape[0],output_alms.shape[-1]):
        output_maps[f,:,c] = hp.alm2map(np.ascontiguousarray(output_alms[f, :, c]), config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
    del output_alms

    if "mask" in compsep_run:
        output_maps[:,compsep_run["mask"] == 0.,:] = 0.

    return output_maps

def _nuiscov_needlet(config: Configs, nuis_alms: np.ndarray, compsep_run: dict, **kwargs) -> None:
    """
    Compute nuisance covariance in the needlet domain.

    Parameters
    ----------
        config: Configs
            Configuration object which includes general settings like nside and lmax. See `compute_and_update_nuisance_covariance` for details.
        nuis_alms: np.ndarray
            Input nuisance alms with shape (n_channels, n_alms).
        compsep_run: dict
            Parameters including needlet configuration for covariance computation. See `compute_and_update_nuisance_covariance` for details.
        **kwargs: dict
            Extra parameters for alm2map/map2alm.

    Returns
    ----------
        None
    """
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    if compsep_run["b_squared"]:
        b_ell = b_ell**2

    if compsep_run['save_needlets']:
        compsep_run["path_out"] = _get_full_path_nuiscov(config, compsep_run)
        os.makedirs(compsep_run["path_out"], exist_ok=True)
        np.save(os.path.join(compsep_run["path_out"], "needlet_bands"), b_ell)
        
    for j in range(b_ell.shape[0]):
        _nuiscov_needlet_j(
            config, nuis_alms, compsep_run,
            b_ell[j], j, **kwargs
        )

    return None

def _fgd_diagnostic_needlet(config: Configs, input_alms: np.ndarray, compsep_run: dict) -> np.ndarray:
    """
    Compute diagnostic maps of foreground emission at different needlet scales.

    Parameters
    ----------
        config (Configs)
            Configuration object which includes general settings like nside and lmax. See `fgd_diagnostic` for details.
        input_alms: np.ndarray
            Input alms with shape (n_channels, n_alms).
        compsep_run: dict
            Dictionary with component separation parameters, including needlet configuration. See `fgd_diagnostic` for details.

    Returns
    -------
        np.ndarray
            Diagnostic needlet-space maps with shape (n_bands, npix).
    """
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    if compsep_run["b_squared"]:
        b_ell = b_ell**2

    if compsep_run['save_needlets']:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        os.makedirs(compsep_run["path_out"], exist_ok=True)
        np.save(os.path.join(compsep_run["path_out"], "needlet_bands"), b_ell)
        
    output_maps = np.zeros((b_ell.shape[0], 12 * config.nside**2))
    for j in range(b_ell.shape[0]):
        output_maps[j] = _fgd_diagnostic_needlet_j(config, input_alms, compsep_run, b_ell[j], noise_debias=compsep_run["cov_noise_debias"][j])
    
    if "mask" in compsep_run:
        output_maps[:,compsep_run["mask"] == 0.] = 0.

    return output_maps

def _gilc_needlet_j(config: Configs, input_alms: np.ndarray, compsep_run: dict,
                    b_ell: np.ndarray, nl_scale: int, depro_cmb: Optional[float] = None, m_bias: Optional[int] = 0, 
                    noise_debias: Optional[float] = 0., **kwargs) -> np.ndarray:
    """
    Perform GNILC component separation in a specific needlet band.

    Parameters
    ----------
    
        config: Configs
            Configuration object which includes general settings like nside, lmax, units. See `gilc` for details.
        input_alms: np.ndarray
            Input alms for the scalar fields. Shape should be (n_channels, n_alms, n_components).
        compsep_run: dict
            Dictionary with component separation parameters. See `gilc` for details.
        b_ell: np.ndarray
            Needlet band window function for the current band. Shape should be (lmax+1,).
        nl_scale : int
            Needlet scale index corresponding to the current GILC run. Used for saving weights with proper label.
        depro_cmb: float, optional
            Deprojection factor for CMB in this needlet band. 
            Residual CMB in GNILC maps will be at the level of depro_CMB * CMB_input.
        m_bias: int, optional
            It will include m_bias more (if m_bias > 0) or less (if m_bias < 0) modes in the reconstructed GNILC maps.
        noise_debias: float, optional
            Noise debiasing factor. If set to a non-zero value, it will subtract a 'noise_debias' fraction of
            noise covariance from the input and nuisance covariance matrices.
        **kwargs: dict
            Additional keyword arguments for healpy function map2alm.

    Returns
    -------
        np.ndarray: 
            Output GNILC alms for the provided needlet band. Shape will be (n_channels_out, n_alms, n_components). 
    """

#    if "mask" in compsep_run or not compsep_run["adapt_nside"]:
#        nside_, lmax_ = config.nside, config.lmax
#    else:
#        nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
    if compsep_run["adapt_nside"]:
        nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax,compsep_run["needlet_config"]["needlet_windows"])
    else:
        nside_, lmax_ = config.nside, config.lmax

    if "mask" in compsep_run:
        if nside_ < config.nside:
            mask = _downgrade_mask(compsep_run["mask"], nside_, threshold=0.)
        else:
            mask = compsep_run["mask"]
    else:
        mask = None
        
    compsep_run["good_channels"] = _get_good_channels_nl(config, b_ell)

    input_maps_nl = np.zeros((compsep_run["good_channels"].shape[0], 12 * nside_**2, input_alms.shape[-1]))
    for n, channel in enumerate(compsep_run["good_channels"]):
        input_alms_j = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        input_maps_nl[n] = np.array([
            hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False)
            for c in range(input_alms.shape[-1])
        ]).T

    output_maps_nl = _gilc_maps(config, input_maps_nl, compsep_run, b_ell, depro_cmb=depro_cmb, 
        m_bias=m_bias, noise_debias=noise_debias, nl_scale=nl_scale, mask=mask)

    del input_maps_nl, mask

    output_alms_j = np.zeros((output_maps_nl.shape[0], hp.Alm.getsize(config.lmax), output_maps_nl.shape[-1]), dtype=complex)
    
    for n in range(output_maps_nl.shape[0]):
        output_alms_nl = np.array([
            hp.map2alm(output_maps_nl[n, :, c], lmax=lmax_, pol=False, **kwargs)
            for c in range(output_maps_nl.shape[-1])
        ]).T
        if compsep_run["b_squared"]:
            output_alms_j[n] = _needlet_filtering(output_alms_nl, np.ones(lmax_+1), config.lmax)
        else:
            output_alms_j[n] = _needlet_filtering(output_alms_nl, b_ell[:lmax_+1], config.lmax)

    del compsep_run['good_channels']
    del output_maps_nl

    return output_alms_j

def _nuiscov_needlet_j(config: Configs, nuis_alms: np.ndarray, compsep_run: dict,
                    b_ell: np.ndarray, nl_scale: int, **kwargs) -> None:
    """
    Perform nuisance covariance computation in a specific needlet band.

    Parameters
    ----------
        config: Configs
            Configuration object which includes general settings like nside, lmax, units. See `compute_and_update_nuisance_covariance` for details.
        nuis_alms: np.ndarray
            Input nuisance alms for the scalar fields. Shape should be (n_channels, n_alms).
        compsep_run: dict
            Dictionary with parameters for nuisance covariance computation. See `compute_and_update_nuisance_covariance` for details.
        b_ell: np.ndarray
            Needlet band window function for the current band. Shape should be (lmax+1,).
        nl_scale : int
            Needlet scale index corresponding to the current run. Used for saving nuisance covariance with proper label.
        **kwargs: dict
            Additional keyword arguments for healpy function map2alm.

    Returns
    -------
        None
    """

#    if "mask" in compsep_run or not compsep_run["adapt_nside"]:
#        nside_, lmax_ = config.nside, config.lmax
#    else:
#        nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
    if compsep_run["adapt_nside"]:
        nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax,compsep_run["needlet_config"]["needlet_windows"])
    else:
        nside_, lmax_ = config.nside, config.lmax   

    if "mask" in compsep_run:
        if nside_ < config.nside:
            mask = _downgrade_mask(compsep_run["mask"], nside_, threshold=0.)
        else:
            mask = compsep_run["mask"]
    else:
        mask = None

    compsep_run["good_channels"] = _get_good_channels_nl(config, b_ell)

    input_maps_nl = np.zeros((compsep_run["good_channels"].shape[0], 12 * nside_**2))
    for n, channel in enumerate(compsep_run["good_channels"]):
        input_alms_j = _needlet_filtering(nuis_alms[channel], b_ell, lmax_)
        input_maps_nl[n] = hp.alm2map(np.ascontiguousarray(input_alms_j), nside_, lmax=lmax_, pol=False)

    _nuiscov(config, input_maps_nl, compsep_run, b_ell, nl_scale=nl_scale, mask=mask)

    return None

def _fgd_diagnostic_needlet_j(config: Configs, input_alms: np.ndarray,
                               compsep_run: dict, b_ell: np.ndarray, noise_debias: Optional[float] = 0.
                            ) -> np.ndarray:
    """
    Compute diagnostic map of foreground complexity for a specific needlet band.

    Parameters
    ----------
        config: Configs
            Configuration object which includes general settings like nside and lmax. See `fgd_diagnostic` for details.
        input_alms: np.ndarray
            Input alms for the scalar fields. Shape should be (n_channels, n_alms).
        compsep_run: dict
            Dictionary with component separation parameters. See `fgd_diagnostic` for details.
        b_ell: np.ndarray
            Needlet band window function for the current band. Shape should be (lmax+1,).
        noise_debias: float, optional
            Noise debiasing factor. If set to a non-zero value, it will subtract a 'noise_debias' fraction of
            noise covariance from the input and nuisance covariance matrices.
    
    Returns
    -------
        np.ndarray
            Diagnostic map of foreground complexity for the provided needlet band.

    """
#    if "mask" in compsep_run or not compsep_run["adapt_nside"]:
#        nside_, lmax_ = config.nside, config.lmax
#    else:
#        nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
    if compsep_run["adapt_nside"]:
        nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax,compsep_run["needlet_config"]["needlet_windows"])
    else:
        nside_, lmax_ = config.nside, config.lmax   

    if "mask" in compsep_run:
        if nside_ < config.nside:
            mask = _downgrade_mask(compsep_run["mask"], nside_, threshold=0.)
        else:
            mask = compsep_run["mask"]
    else:
        mask = None
        
    compsep_run["good_channels"] = _get_good_channels_nl(config, b_ell)
#    compsep_run["good_channels"] = np.arange(input_alms.shape[0])

    input_maps_nl = np.zeros((compsep_run["good_channels"].shape[0], 12 * nside_**2, input_alms.shape[-1]))
    for n, channel in enumerate(compsep_run["good_channels"]):
        input_alms_j = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        input_maps_nl[n] = np.array([
            hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False)
            for c in range(input_alms.shape[-1])
        ]).T

    output_maps_nl = _get_diagnostic_maps(config, input_maps_nl, compsep_run, b_ell, noise_debias=noise_debias, mask=mask)
    del input_maps_nl
    
    if hp.get_nside(output_maps_nl) < config.nside:
        output_maps_nl = hp.ud_grade(output_maps_nl, nside_out=config.nside)

    del compsep_run['good_channels']

    return output_maps_nl

def _gilc_maps(
    config: Configs,
    input_maps: np.ndarray,
    compsep_run: Dict[str, Any],
    b_ell: np.ndarray,
    depro_cmb: Optional[float] = None,
    m_bias: Union[int, float] = 0,
    noise_debias: Optional[float] = 0.,
    nl_scale: Optional[Union[int, None]] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply GILC component separation to the provided scalar multifrequency maps.
    
    Parameters
    ----------
        config: Configs
            Configuration object which includes general settings like nside and lmax. See `gilc` for details.
        input_maps: np.ndarray
            Input maps for the scalar fields. Shape should be (n_channels, npix, n_components).
        compsep_run: Dict[str, Any]
            Dictionary with component separation parameters. See `gilc` for details.
        b_ell: np.ndarray
            Needlet band window function for the current band. Shape should be (lmax+1,).
        depro_cmb: float, optional
            Deprojection factor for CMB in this needlet band. 
            Residual CMB in GNILC maps will be at the level of depro_CMB * CMB_input.
        m_bias: int, optional
            It will include m_bias more (if m_bias > 0) or less (if m_bias < 0) modes in the reconstructed GNILC maps.
        noise_debias: float, optional
            Noise debiasing factor. If set to a non-zero value, it will subtract a 'noise_debias' fraction of
            noise covariance from the input and nuisance covariance matrices.
        nl_scale : int, optional
            Needlet scale index corresponding to the current GILC run. Used for saving weights with proper label.
        mask: np.ndarray, optional
            Optional mask to be applied to the input maps. Shape should be (npix,).
            If non-binary, it will be applied as a weight during covariance computation.
    
    Returns
    -------
        np.ndarray
            Output GNILC maps with shape (n_channels_out, npix, n_components).
    
    """
    # Compute covariance matrices for input and nuisance maps
    cov = (get_ilc_cov(input_maps[...,0], config.lmax, compsep_run, config.fwhm_out, b_ell, mask=mask)).T
    if compsep_run["ilc_bias"] != 0. and mask is not None:
        mask_cov = (cov[:,0,0] != 0.).astype(float)
        if np.sum(mask_cov == 0.) == mask_cov.shape[0]:
            raise ValueError("The provided mask masks out all the pixels used for covariance computation.")
        elif np.sum(mask_cov == 0.) == 0:
            mask_cov = None
    else:
        mask_cov = None
    
    if not ("load_nuisance_covariance" in compsep_run and compsep_run["load_nuisance_covariance"]):
        if isinstance(compsep_run["nuis_idx"], int):
            cov_n = (get_ilc_cov(input_maps[...,compsep_run["nuis_idx"]], config.lmax, compsep_run, config.fwhm_out, b_ell, mask=mask)).T
        elif isinstance(compsep_run["nuis_idx"], list):
            nuis_maps = np.zeros_like(input_maps[...,0])
            for idx in compsep_run["nuis_idx"]:
                nuis_maps += input_maps[...,idx]
            cov_n = (get_ilc_cov(nuis_maps, config.lmax, compsep_run, config.fwhm_out, b_ell, mask=mask)).T
            del nuis_maps
    else:
        path_nuiscov = _get_full_path_nuiscov(config, compsep_run)
        cov_n = load_nuiscov(config, path_nuiscov, compsep_run,
                            hp.npix2nside(input_maps.shape[-2]), compsep_run["nuisance"], nl_scale=nl_scale)

    if noise_debias != 0.:
        if not ("load_noise_covariance" in compsep_run and compsep_run["load_noise_covariance"]):
            cov_noi = (get_ilc_cov(input_maps[...,compsep_run["noise_idx"]], config.lmax, compsep_run, config.fwhm_out, b_ell, mask=mask)).T
        else:
            path_nuiscov = _get_full_path_nuiscov(config, compsep_run)
            cov_noi = load_nuiscov(config, path_nuiscov, compsep_run,
                                hp.npix2nside(input_maps.shape[-2]), "noise", nl_scale=nl_scale)

        cov = cov - noise_debias * cov_noi
        cov_n = cov_n - noise_debias * cov_noi
        del cov_noi

    if mask_cov is not None:
        cov[mask_cov == 0.,...] = np.repeat(np.mean(cov[mask_cov != 0.,...], axis=0, keepdims=True), np.sum(mask_cov == 0.), axis=0)
        cov_n[mask_cov == 0.,...] = np.repeat(np.mean(cov_n[mask_cov != 0.,...], axis=0, keepdims=True), np.sum(mask_cov == 0.), axis=0)
        mask_cov = None

    if mask_cov is not None:
        λ, U = Cn_C_Cn(cov[mask_cov != 0.],cov_n[mask_cov != 0.])
    else:
        λ, U = Cn_C_Cn(cov,cov_n)

    λ[λ<1.]=1.

    W = _get_gilc_weights(config, U, λ, cov, cov_n, input_maps.shape, compsep_run, depro_cmb=depro_cmb, m_bias=m_bias, mask_cov=mask_cov)
    del mask_cov, U, λ
#    if compsep_run["ilc_bias"] != 0. and mask is not None:
#        W[mask == 0.,...] = 0.

    if compsep_run["save_weights"]:
        compsep_run["path_out"] = _get_full_path_out(config, compsep_run)
        save_ilc_weights(config, W, compsep_run,
                         hp.npix2nside(input_maps.shape[-2]), nl_scale=nl_scale)

    del cov, cov_n

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

def _nuiscov(
    config: Configs,
    input_maps: np.ndarray,
    compsep_run: Dict[str, Any],
    b_ell: np.ndarray,
    nl_scale: Optional[Union[int, None]] = None,
    mask: Optional[np.ndarray] = None,
) -> None:
    """
    Compute nuisance covariance for the provided scalar multifrequency maps.
    
    Parameters
    ----------
        config: Configs
            Configuration object which includes general settings like nside and lmax. See `compute_and_update_nuisance_covariance` for details.
        input_maps: np.ndarray
            Input nuisance maps for the scalar fields. Shape should be (n_channels, npix).
        compsep_run: Dict[str, Any]
            Dictionary with settings for nuisance covariance computation. See `compute_and_update_nuisance_covariance` for details.
        b_ell: np.ndarray
            Needlet band window function for the current band. Shape should be (lmax+1,).
        nl_scale : int, optional
            Needlet scale index corresponding to the provided input maps. Used for saving covariance with proper label.
        mask: np.ndarray, optional
            Optional mask to be applied to the input maps. Shape should be (npix,).
            If non-binary, it will be applied as a weight during covariance computation.
    
    Returns
    -------
        None
    
    """
    # Compute covariance matrices for input and nuisance maps
    cov_n = (get_ilc_cov(input_maps, config.lmax, compsep_run, config.fwhm_out, b_ell, mask=mask)).T
    
    compsep_run["path_out"] = _get_full_path_nuiscov(config, compsep_run)
    update_and_save_nuiscov_serial(config, cov_n, compsep_run,
                        hp.npix2nside(input_maps.shape[-1]), nl_scale=nl_scale)

    return None

def _get_diagnostic_maps(
    config: Configs,
    input_maps: np.ndarray,
    compsep_run: dict,
    b_ell: np.ndarray,
    noise_debias: Optional[float] = 0.,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Get diagnostic maps of foreground complexity for provided scalar field either in pixel or needlet domain.

    Parameters
    ----------
        config: Configs
            Configuration object which includes general settings like nside and lmax. See `fgd_diagnostic` for details.
        input_maps: np.ndarray
            Input maps for the scalar fields. Shape should be (n_channels, npix, n_comps).
        compsep_run: dict
            Dictionary with component separation parameters. See `fgd_diagnostic` for details.
        b_ell: np.ndarray
            Needlet band window function for the current band. Shape should be (lmax+1,).
        noise_debias: float, optional
            Noise debiasing factor. If set to a non-zero value, it will subtract a 'noise_debias' fraction of
            noise covariance from the input and nuisance covariance matrices.
        mask: np.ndarray, optional
            Optional mask to be applied to the input maps. Shape should be (npix,).
            If non-binary, it will be applied as a weight during covariance computation.
    
    Returns
    -------
        np.ndarray
            Diagnostic map of foreground complexity for the provided scalar field.

    """

    cov = (get_ilc_cov(input_maps[...,0], config.lmax, compsep_run, config.fwhm_out, b_ell, mask=mask)).T
    if compsep_run["ilc_bias"] != 0. and mask is not None:
        mask_cov = (cov[:,0,0] != 0.).astype(float)
        if np.sum(mask_cov == 0.) == mask_cov.shape[0]:
            raise ValueError("The provided mask masks out all the pixels used for covariance computation.")
        elif np.sum(mask_cov == 0.) == 0:
            mask_cov = None
    else:
        mask_cov = None

    if not ("load_nuisance_covariance" in compsep_run and compsep_run["load_nuisance_covariance"]):
        cov_n = (get_ilc_cov(input_maps[...,compsep_run["nuis_idx"]], config.lmax, compsep_run, config.fwhm_out, b_ell, mask=mask)).T
    else:
        path_nuiscov = _get_full_path_nuiscov(config, compsep_run)
        cov_n = load_nuiscov(config, path_nuiscov, compsep_run,
                            hp.npix2nside(input_maps.shape[-2]), compsep_run["nuisance"], nl_scale=nl_scale)

    if noise_debias != 0.:
        if not ("load_noise_covariance" in compsep_run and compsep_run["load_noise_covariance"]):
            cov_noi = (get_ilc_cov(input_maps[...,compsep_run["noise_idx"]], config.lmax, compsep_run, config.fwhm_out, b_ell, mask=mask)).T
        else:
            path_nuiscov = _get_full_path_nuiscov(config, compsep_run)
            cov_noi = load_nuiscov(config, path_nuiscov, compsep_run,
                            hp.npix2nside(input_maps.shape[-2]), "noise", nl_scale=nl_scale)
        cov = cov - noise_debias * cov_noi
        cov_n = cov_n - noise_debias * cov_noi
        del cov_noi

    if mask_cov is not None:
        λ, U = Cn_C_Cn(cov[mask_cov != 0.],cov_n[mask_cov != 0.])
    else:
        λ, U = Cn_C_Cn(cov,cov_n)
    λ[λ<1.]=1.
    del cov, cov_n, U

    m = _get_gilc_m(λ) 

    if isinstance(m, (int, float)):
        m = np.repeat(m, input_maps.shape[-2])
#        if "mask" in compsep_run:
#            m[compsep_run["mask"] == 0.] = 0.
        if mask is not None:
            m[mask == 0.] = 0.
        return m
    else:
#        if "mask" in compsep_run:
#            m_full = np.zeros(input_maps.shape[-2])
#            m_full[compsep_run["mask"] > 0.] = m
#            return m_full
        if mask is not None:
            if mask_cov is None:
                m_full = hp.upgrade(m, nside_out=npix2nside(input_maps.shape[-2]))
                m_full[mask == 0.] = 0.
            else:
                m_ = np.zeros(mask_cov.shape[0])
                m_[mask_cov != 0.] = m
                del m
                m_full = hp.ud_grade(m_, nside_out=npix2nside(input_maps.shape[-2]))
                m_full[mask == 0.] = 0.
            return m_full
        else:
            return m

def _get_gilc_weights(
    config: Configs,
    U: np.ndarray,
    λ: np.ndarray,
    cov: np.ndarray,
    cov_n: np.ndarray,
    input_shapes: tuple,
    compsep_run: dict,
    depro_cmb: Optional[float] = None,
    m_bias: Union[int, float] = 0,
    mask_cov: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Get GILC weights for the provided scalar field.

    Parameters
    ----------
        config: Configs
            Configuration object which includes general settings like nside and lmax. See `gilc` for details.
        U: np.ndarray
            Eigenvectors of the covariance matrix. Shape should be (n_channels, n_channels, n_pix).
        λ: np.ndarray
            Eigenvalues of the covariance matrix. Shape should be (n_channels, n_pix).
        cov: np.ndarray
            Covariance matrix of the input maps. Shape should be (n_channels, n_channels, n_pix).
        cov_n: np.ndarray
            Covariance matrix of the nuisance maps. Shape should be (n_channels, n_channels, n_pix).
        input_shapes: tuple
            Shapes of the input maps, typically (n_channels, npix, n_components).
        compsep_run: dict
            Dictionary with component separation parameters. See `gilc` for details.
        depro_cmb: float, optional
            Deprojection factor for CMB in this needlet band.
            Residual CMB in GNILC maps will be at the level of depro_CMB * CMB_input.
        m_bias: int, optional
            It will include m_bias more (if m_bias > 0) or less (if m_bias < 0) modes in the reconstructed GNILC maps.
        mask_cov: np.ndarray, optional
            Optional mask applied during covariance computation. Shape should be (npix,).
    
    Returns
    -------
        np.ndarray
            GILC weights for the provided scalar field. 
            Shape will be (n_channels, n_channels) if cov is 2D, or (npix, n_channels, n_channels) if cov is 3D.
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
            W = F_e @ lg.inv(F_A.T @ inv_cov @ F_A) @ F_A.T @ inv_cov

    elif cov.ndim == 3:
        m = _get_gilc_m(λ)
        m += int(m_bias)  
        del λ

        covn_sqrt = lg.cholesky(cov_n)
#        covn_sqrt_inv = lg.inv(covn_sqrt)
        del cov_n
        
        W_=np.zeros((cov.shape[0],input_shapes[0],input_shapes[0]))

        for m_ in np.unique(m):
            U_s = U[m==m_,:,:m_]
            cov_inv = lg.inv(cov[m==m_])
            F = np.einsum("kij,kjz->kiz", covn_sqrt[m==m_], U_s)

            if depro_cmb is None: 
                W_[m==m_] = np.einsum(
                    "kil,klj->kij", F,
                    np.einsum("kzl,klj->kzj",
                              lg.inv(np.einsum("kiz,kij,kjl->kzl", F, cov_inv, F)),
                              np.einsum("kiz,kij->kzj", F, cov_inv))
                )
            else:
                #e_cmb = depro_cmb * np.ones((F.shape[0],F.shape[1],1)) 
                #F_e = np.concatenate((F,e_cmb),axis=2)
                F_e = np.concatenate((F,np.tile(depro_cmb * A_cmb, (F.shape[0], 1))[:, :, np.newaxis]),axis=2)
                F_A = np.concatenate((F,np.tile(A_cmb, (F.shape[0], 1))[:, :, np.newaxis]),axis=2)
                W_[m==m_] = np.einsum(
                    "kil,klj->kij", F_e,
                    np.einsum("kzl,klj->kzj",
                              lg.inv(np.einsum("kiz,kij,kjl->kzl", F_A, cov_inv, F_A)),
                              np.einsum("kiz,kij->kzj", F_A, cov_inv))
                )
                del F_e, F_A #e_cmb, 

        del covn_sqrt, cov_inv, U_s, F, m

#        if "mask" in compsep_run:
#            W = np.zeros((input_shapes[1],W_.shape[1],W_.shape[2]))
#            W[compsep_run["mask"] > 0.] = np.copy(W_)
        if mask_cov is not None:
            W__ = np.zeros((mask_cov.shape[0],W_.shape[1],W_.shape[2]))
            W__[mask_cov != 0.] = np.copy(W_)
            del W_
            if W__.shape[0] != input_shapes[1]:
                W = np.zeros((input_shapes[1],W__.shape[1],W__.shape[2]))
                for i, k in np.ndindex(W__.shape[1],W__.shape[2]):
                    W[:,i,k]=hp.ud_grade(W__[:,i,k],hp.npix2nside(input_shapes[1]))
            else:
                W=np.copy(W__)
            del W__
        else:
            if W_.shape[0] != input_shapes[1]:
                W = np.zeros((input_shapes[1],W_.shape[1],W_.shape[2]))
                for i, k in np.ndindex(W_.shape[1],W_.shape[2]):
                    W[:,i,k]=hp.ud_grade(W_[:,i,k],hp.npix2nside(input_shapes[1]))
            else:
                W=np.copy(W_)
            del W_

    return W

def _get_gilc_m(λ: np.ndarray) -> Union[int, np.ndarray]:
    """
    Estimate the number of signal components using eigenvalue-based AIC.

    Parameters
    ----------
        λ: np.ndarray
            Eigenvalues of the whitened signal+nuisance covariance matrix.

    Returns
    -------
        int or np.ndarray
            Estimated number of components. If λ is 1D, returns a single integer.
            If λ is 2D, returns an array of integers of shape (npix).
    """

    if λ.ndim==1:
        A_m=np.zeros(λ.shape[0]+1)
        for i in range(λ.shape[0]):
            A_m[i]=2*i + np.sum(λ[i:]-np.log(λ[i:])-1.)
        A_m[λ.shape[0]]=2.*λ.shape[0]
        m = int(np.argmin(A_m))
    elif λ.ndim==2:
        A_m=np.zeros((λ.shape[0],λ.shape[1]+1))
        for i in range(λ.shape[1]):
            A_m[:,i]=2*i + np.sum(λ[:,i:]-np.log(λ[:,i:])-1.,axis=1)
        A_m[:,λ.shape[1]]=2.*λ.shape[1]
        m = np.argmin(A_m,axis=1)
    
    del A_m

    return m

def Cn_C_Cn(C: np.ndarray, C_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Whiten the signal covariance with nuisance and perform SVD.

    Parameters
    ----------
        C: np.ndarray
            Signal covariance matrix. Shape should be (n_channels, n_channels) or (npix, n_channels, n_channels).
        C_n: np.ndarray
            Nuisance covariance matrix. Shape should be (n_channels, n_channels) or (npix, n_channels, n_channels).

    Returns
    -------
        tuple[np.ndarray, np.ndarray]
            Tuple of eigenvalues and eigenvectors.
    """
    if (C.ndim == 2) and (C_n.ndim == 2):
        Cn_sqrt = lg.inv(scipy.linalg.sqrtm(C_n))
        M = lg.multi_dot([Cn_sqrt,C,Cn_sqrt])
    elif (C.ndim == 3) and (C_n.ndim == 3):
        Cn_sqrt = lg.inv(lg.cholesky(C_n))
        M = np.einsum("kij,kjz->kiz", Cn_sqrt, np.einsum("kij,kzj->kiz", C, Cn_sqrt))
    U, λ, _ = lg.svd(M)
    return λ, U

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]
                    

   
    