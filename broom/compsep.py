import warnings
from functools import partialmethod
from typing import Optional, Tuple

import numpy as np
import healpy as hp
import os
from types import SimpleNamespace
import sys

from .configurations import Configs
from .routines import (merge_dicts, _map2alm_kwargs, _slice_data, _format_nsim, _log,   
            _EB_to_QU, _QU_to_EB, _B_to_QU, _E_to_QU)
from .inputs import _alms_from_data
from .gilcs import gilc, fgd_diagnostic
from .ilcs import ilc
from .pilcs import pilc
from .gpilcs import gpilc, fgd_P_diagnostic
from .templates import get_residuals_template
from .masking import get_masks_for_compsep
from .saving import get_gnilc_maps
from .needlets import _get_needlet_windows_
from typing import Dict, Any, Union


def get_data_and_compsep(config: Configs, foregrounds, nsim=None):
    """
    Get data and run component separation based on the provided configuration.

    Parameters
    ----------
        config : Configs
            Configuration object with settings for component separation and data handling.
            See 'component_separation' for details on required attributes.
        
        foregrounds : SimpleNamespace
            Foreground data to be used in component separation. 
            It should have the 'total' attribute, a numpy array of shape (n_channels, 3, n_alms/n_pixs)
            where 3 correspond to (T, Q, U) if data_type is "maps" or (T, E, B) if data_type is "alms".

        nsim : Optional[int or str], optional
            Simulation number to be used for saving the outputs. If None, the outputs will be saved without label on simulation number.
            Default: None.
    
    Returns
    -------
        SimpleNamespace or None
            If `config.return_compsep_products` is True, returns outputs. Otherwise returns None.
    
    Raises
    ------
        ValueError
            If the configuration is invalid or required fields are missing.
    """

    from broom import _get_data_simulations_  # Importing here to avoid circular import issues
    data = _get_data_simulations_(config, foregrounds, nsim=nsim)

    if config.return_compsep_products:
        outputs = SimpleNamespace()
        outputs_ = component_separation(config, data, nsim=nsim)
        for attr in vars(outputs_):
            setattr(outputs, attr, getattr(outputs_, attr))
        return outputs
    else:
        component_separation(config, data, nsim=nsim)
        return None

def component_separation(config: Configs, data: SimpleNamespace, nsim: Optional[Union[int, str]] = None, **kwargs) -> Optional[SimpleNamespace]:  # shape (N_sims, lmax - 1)
    r"""
    Run component separation methods on input data based on the specified configuration.

    Parameters
    ----------
        config : Configs
            Configuration object containing the settings for component separation.
            In particular, it should include the following attributes:
            - data_type (str): Type of data, either "maps" or "alms".
            - field_in (str): Field associated to the provided data, e.g. TQU, QU; TEB; EB.
            - field_out (str): Fields of the desired outputs from component separation. Default: `field_in`.
            - lmax (int): Desired maximum multipole for the component separation and output products.
            - nside (int): Desired HEALPix resolution of the outputs.
            - fwhm_out (float): Full width at half maximum of the output maps in arcminutes.
            - compsep (Dict[str, Any]): List of dictionaries containing the configuration for each component separation method to be run.
            - mask_observations (str): Path to HEALPix mask fits file, if any. Default: None. 
                It is used to exclude unobserved regions in the alm computation and component separation.
            - mask_covariance (str): Full path to mask used to weight adn/or exclude pixels in component separation. Default: None.
            - leakage_correction (str): Whether to apply EB-leakage correction on input data if mask_type is "observed_patch". Default: None.
            - bring_to_common_resolution (bool): Whether to bring the data to a common angular resolution (correcting for input beams). If False, the data will be used as is. Default: True.
            - pixel_window_in (bool): Whether pixel window is included in the input data. Default: False.
            - pixel_window_out (bool): Whether to include pixel window in the output products. Default: False.
            - save_compsep_products (bool): Whether to save the outputs of component separation. Default: True.
            - return_compsep_products (bool): Whether to return the outputs of component separation. Default: False.
            - path_outputs (str): Path to the directory where the outputs will be saved if 'save_compsep_products' is True. 
                        It will save them in "working_directory/{path_outputs}/{method_specs}", where {method_specs} is a string containing the method name, domain, and other relevant parameters.
                        Default: Working directory + "/outputs/".
            - verbose (bool): Whether to print information about the component separation process. Default: False.
    
        data : SimpleNamespace
            Data object containing the input data for component separation. It should have the following attributes:
            - `total`: Total map or alms to be used for component separation.
            - Other optional attributes such as `noise`, `cmb`, `fgds`, etc.

        nsim : Optional[int or str], optional
            Simulation number to be used for saving the outputs. If None, the outputs will be saved without label on simulation number.
            Default: None.

    Returns
    -------
        SimpleNamespace or None
            If `config.return_compsep_products` is True, returns outputs. Otherwise returns None.
    """

    # Initializing nsim
    nsim = _format_nsim(nsim)
    if nsim is None:
        _log(f"Simulation number not provided. If 'save_compsep_products' is set to True, the outputs will be saved without label on simulation number.", verbose=config.verbose)
            
    kwargs = _map2alm_kwargs(**kwargs)

    # Validation of config parameters and data format
    config = _check_data_and_config(config, data)
    config = _check_fields(config, data)

    # Slicing data if necessary
    if (data.total.ndim > 2) and (config.field_out != config.field_in):
        config.field_in_cs = _get_field_in_cs(config.field_in, config.field_out)
        data = _slice_data(data, config.field_in, config.field_in_cs)
    else:
        config.field_in_cs = config.field_in

    if nsim is None:
        msg = f"Computing required input alms for component separation." if config.data_type=="maps" else f"Pre-processing input alms for component separation."    
    else:
        msg = f"Computing required input alms for component separation for simulation {nsim}." if config.data_type=="maps" else f"Pre-processing input alms for component separation for simulation {nsim}."
    _log(msg, verbose=config.verbose)

    # Preprocessing arguments for alms computation
    preprocess_args = dict(
            data_type=config.data_type, 
            bring_to_common_resolution=config.bring_to_common_resolution, 
            pixel_window_in=config.pixel_window_in, 
            **kwargs)

    # Initializing mask
#    if config.mask_path is not None:
#        if not hasattr(config, "mask_type"):
#            config.mask_type = "mask_for_compsep"
#        if config.mask_type not in ["mask_for_compsep", "observed_patch"]:
#            raise ValueError("Invalid value for 'mask_type'. It can be either 'mask_for_compsep' or 'observed_patch'.")
#        else:
#            _log(f"Provided mask is used as" + " observed patch" if config.mask_type == "observed_patch" else " mask for component separation.", verbose=config.verbose)
#
#        if not isinstance(config.mask_path, str):
#            raise ValueError("Invalid mask_path in config. It must be a string full path to a HEALPix mask fits file.")#
#
#        mask = hp.read_map(config.mask_path, field=0)
#
#        if (config.mask_type == "observed_patch") and (config.data_type == "maps"):
#            preprocess_args["mask_in"] = _preprocess_mask(mask, config.nside_in)
#    else:
#        mask = None
    mask_obs, mask_cov = get_masks_for_compsep(config.mask_observations, config.mask_covariance, config.nside_in)
    preprocess_args["mask_in"] = np.copy(mask_obs) if mask_obs is not None else None

    # Computing alms from input data
    input_alms = _alms_from_data(config, data, config.field_in_cs, **preprocess_args)

    methods_map = {
    'ilc': ilc,
    'gilc': gilc,
    'cilc': ilc,
    'c_ilc': ilc,
    'mc_ilc': ilc,
    'mc_cilc': ilc,
    'mcilc': ilc,
    'pilc': pilc,
    'cpilc': pilc,
    'c_pilc': pilc,
    'gpilc': gpilc,
    'fgd_diagnostic': fgd_diagnostic,
    'fgd_P_diagnostic': fgd_P_diagnostic
    }
    #}

    # Initializing outputs if required
    if config.return_compsep_products:
        outputs = SimpleNamespace()
        if any(compsep_run["method"] not in ["fgd_diagnostic", "fgd_P_diagnostic"] for compsep_run in config.compsep):
        # Initialize attributes of `outputs` with empty lists
            for attr in vars(data):
                setattr(outputs, attr, [])
        
    _log(f"Running component separation for simulation {nsim}." if nsim is not None else f"Running component separation.", verbose=config.verbose)

    mask_obs, mask_cov = get_masks_for_compsep(mask_obs, mask_cov, config.nside)
    
    # Starting component separation runs
    for compsep_run in config.compsep:
        compsep_run = _standardize_compsep_config(compsep_run, config.lmax, save_products=config.save_compsep_products)
        compsep_run["nsim"] = nsim

#        if mask is not None:
#            compsep_run["mask"] = _preprocess_mask(mask, config.nside)
        if mask_cov is not None:
            compsep_run["mask"] = mask_cov

        _log(f"Running {compsep_run['method']} in {compsep_run['domain']} domain for simulation {nsim}." if nsim is not None else f"Running {compsep_run['method']} in {compsep_run['domain']} domain.", verbose=config.verbose)
        
        if config.return_compsep_products:
            prod = methods_map[compsep_run["method"]](config, input_alms, compsep_run, **kwargs)
            if compsep_run["method"] in ["fgd_diagnostic", "fgd_P_diagnostic"]:
                if hasattr(outputs, 'm'):
                    getattr(outputs, 'm').append(getattr(prod, 'm'))
                else:
                    setattr(outputs, 'm', [])
                    getattr(outputs, 'm').append(getattr(prod, 'm'))
            else:
                for attr in vars(prod).keys():
                    getattr(outputs, attr).append(getattr(prod, attr))
        else:
            methods_map[compsep_run["method"]](config, input_alms, compsep_run, **kwargs)

        compsep_run.pop("mask", None)
        del compsep_run["nsim"]
    
    del mask_obs, mask_cov

    if config.return_compsep_products:
#        for attr in vars(outputs).keys():
#            setattr(outputs, attr, np.array(getattr(outputs, attr)))
        return outputs
    
    return None

def estimate_residuals(config: Configs, nsim: Optional[Union[int, str]] = None, **kwargs) -> Optional[SimpleNamespace]:  
    """
    Estimate residual foregrounds from component-separated maps.

    Parameters
    ----------
        config : Configs
            Configuration object with settings for component separation. It should include:
            - mask_observations (str): Full path to HEALPix mask fits file, if any. Default: None.
                        It is used to exclude unobserved regions from loaded foreground tracers.
            - mask_covariance (str): Full path to mask used to weight pixels in component separation.
            - field_out (str): Fields of the desired outputs for residuals estimate. Default: `config.field_in`.
            - compsep_residuals (list): List of dictionaries with settings for each component separation method to estimate residuals. It should contain:
                - compsep_path (str): Path to the directory containing the 'weights' folder of corresponding component separation method.
                                    Weights will be loaded from "working_directory/{compsep_path}/weights. 
                - gnilc_path (str): Path to the directory containing the 'gnilc' maps to be used as multifrequency foreground tracers.
                                    Tracers will be loaded from "working_directory/{gnilc_path}/output_total".
                - field_in (str): Fields of the data to be loaded. Optional, default is `config.field_out`.
            - lmax (int): Maximum multipole for output products.
            - nside (int): HEALPix resolution associated to GNILC tracers and component separation weights.
            - fwhm_out (float): Full width at half maximum associated to GNILC tracer and component separation products.
            - pixel_window_out (bool): Whether to include pixel window in the output products. Default: False.
            - return_compsep_products (bool): Whether to return the residuals estimate. Default: False.
            - save_compsep_products (bool): Whether to save the residuals estimate. Default: True.
            - verbose (bool): Whether to print information about the residuals estimate. Default: False.

        nsim : int | str | None
            Simulation number or identifier.

        **kwargs
            Additional keyword arguments for internal helper functions.

    Returns
    -------
        SimpleNamespace or None
            Residual products if `config.return_compsep_products` is True, otherwise None.
    """

    # Initializing nsim
    nsim = _format_nsim(nsim)
    if nsim is None:
        _log(f"Simulation number not provided. If 'save_compsep_products' is set to True, the outputs will be saved without label on simulation number.", verbose=config.verbose)        

    kwargs = _map2alm_kwargs(**kwargs)

    # Initializing mask
#    if config.mask_path is not None:
#        if not hasattr(config, "mask_type"):
#            config.mask_type = "mask_for_compsep"
#        if config.mask_type not in ["mask_for_compsep", "observed_patch"]:
#            raise ValueError("Invalid value for 'mask_type'. It can be either 'mask_for_compsep' or 'observed_patch'.")
#        else:
#            _log(f"Provided mask is used as" + "observed patch" if config.mask_type == "observed_patch" else "mask for component separation.", verbose=config.verbose)
#        mask = hp.read_map(config.mask_path, field=0)
#    else:
#        mask = None

    _log(f"Running foregrounds residuals estimation for simulation {nsim}." if nsim is not None else f"Running component separation.", verbose=config.verbose)

    outputs = SimpleNamespace() if config.return_compsep_products else None

    # Setting field_out if not provided
    if not config.field_out:
        config.field_out = config.field_in

    nside_in_in = config.nside_in
    config.nside_in = config.nside

    mask_obs, mask_cov = get_masks_for_compsep(config.mask_observations, config.mask_covariance, config.nside)

    # Starting loop for estimation of foreground residuals for the requested cases
    for compsep_run in config.compsep_residuals:
        delete_field_in = "field_in" not in compsep_run
        compsep_run.setdefault("field_in", config.field_out)
        compsep_run["nsim"] = nsim
        compsep_run.setdefault("adapt_nside", False)

        data = get_gnilc_maps(config, compsep_run["gnilc_path"], field_in=compsep_run["field_in"], nsim=nsim)
        
        if (data.total.ndim > 2) and (config.field_out != compsep_run["field_in"]):
            compsep_run["field_in_cs"] = _get_field_in_cs(compsep_run["field_in"], config.field_out)
            data = _slice_data(data, compsep_run["field_in"], compsep_run["field_in_cs"])
        else:
            compsep_run["field_in_cs"] = compsep_run["field_in"]

        if config.return_compsep_products:
            for attr in vars(data):
                if not hasattr(outputs, attr):
                    setattr(outputs, attr, [])

        preprocess_args = dict(data_type="maps",
            bring_to_common_resolution=False,
            pixel_window_in=config.pixel_window_out,
            **kwargs
        )

#        if mask is not None:
#            if (config.mask_type == "observed_patch"):
#                 preprocess_args["mask_in"] = _preprocess_mask(mask, config.nside_in)
        if mask_obs is not None:
            preprocess_args["mask_in"] = mask_obs

        input_alms = _alms_from_data(config, data, compsep_run["field_in_cs"], **preprocess_args)
        
#        if mask is not None:
#            compsep_run["mask"] = _preprocess_mask(mask, config.nside)
        if mask_cov is not None:
            compsep_run["mask"] = mask_cov

        if config.return_compsep_products:
            prod = get_residuals_template(config, input_alms, compsep_run, **kwargs)
            for attr in vars(prod).keys():
                getattr(outputs, attr).append(getattr(prod, attr))
        else:
            get_residuals_template(config, input_alms, compsep_run, **kwargs)

        compsep_run.pop("mask", None)
        del compsep_run["nsim"]
        if delete_field_in:
            del compsep_run["field_in"]

    config.nside_in = nside_in_in

    if config.return_compsep_products:
        for attr in vars(outputs).keys():
            setattr(outputs, attr, np.array(getattr(outputs, attr)))
        return outputs
    return None

def _standardize_compsep_config(compsep_run: Dict[str, Any], lmax: int, save_products: bool = True) -> Dict[str, Any]:
    """
    Standardize and validate the component separation configuration dictionary.

    Parameters
    ----------
        compsep_run : dict
            Dictionary with method settings for component separation.

        lmax : int
            Maximum multipole for the component separation and output products.

        save_products : bool, optional
            If True, ancillary outputs like needlets or weights will be saved.

    Returns
    -------
        dict
            Updated and validated component separation configuration dictionary.
    """

    if compsep_run["domain"] == "pixel":
        compsep_run["b_squared"] = False 
        compsep_run["adapt_nside"] = False 
    if compsep_run["domain"] == "needlet":
        # Update the original dictionary with the merged dictionary
        if "needlet_config" not in compsep_run:
            raise ValueError("needlet_config must be provided if compsep domain is needlet.")
        compsep_run['needlet_config'] = merge_dicts(compsep_run['needlet_config'])
        compsep_run.setdefault("b_squared", False)
        compsep_run.setdefault("adapt_nside", True) 
        compsep_run.setdefault("save_needlets", save_products)

    if compsep_run["method"] != "mcilc":
        compsep_run.setdefault("ilc_bias", 0.)
        nls_number = _get_needlet_windows_(compsep_run["needlet_config"], lmax).shape[0] if compsep_run["domain"] == "needlet" else 1
        compsep_run.setdefault("cov_noise_debias", np.zeros(nls_number)) if nls_number > 1 else compsep_run.setdefault("cov_noise_debias", 0.)
        if compsep_run["domain"]=="pixel":
            if isinstance(compsep_run["cov_noise_debias"], (list, np.ndarray)):
                raise ValueError("cov_noise_debias must be a scalar when domain is pixel.")
        elif compsep_run["domain"]=="needlet":
            if isinstance(compsep_run["cov_noise_debias"], (float, int)):
                compsep_run["cov_noise_debias"] = np.repeat(compsep_run["cov_noise_debias"], nls_number)
            elif isinstance(compsep_run["cov_noise_debias"], list):
                compsep_run["cov_noise_debias"] = (compsep_run["cov_noise_debias"] + [0.] * nls_number)[:nls_number]
            elif isinstance(compsep_run["cov_noise_debias"], np.ndarray):
                _len = nls_number - compsep_run["cov_noise_debias"].shape[0]
                if _len > 0:
                    compsep_run["cov_noise_debias"] = np.append(compsep_run["cov_noise_debias"], [0.] * _len)
                compsep_run["cov_noise_debias"] = compsep_run["cov_noise_debias"][:nls_number]    
            else:
                raise ValueError("cov_noise_debias must be a scalar, a list or a np.ndarray if domain is needlet.")

    if compsep_run["domain"] in ["pixel", "needlet"]:
        if compsep_run["method"] != "mcilc":
            compsep_run.setdefault("reduce_ilc_bias", False)
        
    if compsep_run["method"] in ["c_ilc","c_pilc","mc_ilc","mc_cilc"]:
        if compsep_run["domain"] != "needlet":
            raise ValueError("The methods 'c_ilc', 'c_pilc', 'mc_ilc' and 'mc_cilc' can only be used in the needlet domain.")
        if "special_nls" not in compsep_run:
            raise ValueError("special_nls must be provided for methods 'c_ilc', 'c_pilc', 'mc_ilc' and 'mc_cilc'.")      
        if not isinstance(compsep_run["special_nls"], list):
            raise ValueError("special_nls must be a list of integers.")
        
    if compsep_run["method"] in ["cilc", "c_ilc", "mc_cilc", "cpilc", "c_pilc"]:
        if "constraints" not in compsep_run:
            raise ValueError("A dictionary of constraints must be provided in the compsep_run dictionary for methods 'cilc', 'c_ilc', 'cpilc', 'c_pilc', 'mc_cilc'.")
        compsep_run['constraints'] = merge_dicts(compsep_run['constraints'])
    
    if compsep_run["method"] in ["mcilc", "mc_ilc", "mc_cilc"]:
        compsep_run.setdefault("mc_type", "cea_real")
        if compsep_run["mc_type"] not in ["cea_ideal","cea_real","rp_ideal","rp_real"]:
            raise ValueError("Invalid value for mc_type. It must be 'cea_ideal', 'cea_real', 'rp_ideal' or 'rp_real'.")

        if "real" in compsep_run["mc_type"]:
            if "path_tracers" not in compsep_run or not isinstance(compsep_run["path_tracers"], str):
                raise ValueError("Path to tracers ('path_tracers') must be provided and a string for methods 'mcilc' and 'mc_ilc'.")
            compsep_run["path_tracers"] = compsep_run["path_tracers"] if compsep_run["path_tracers"].endswith('/') else compsep_run["path_tracers"] + '/'

        if "channels_tracers" not in compsep_run:
            raise ValueError("channels_tracers must be provided for methods 'mcilc' and 'mc_ilc'. It must be a list of two integers corresponding to the indices of the channels you want to use for the tracer.")
        if (not isinstance(compsep_run["channels_tracers"], list)) or (len(compsep_run["channels_tracers"]) != 2):
            raise ValueError("channels_tracers must be a list of two integers corresponding to the indices of the channels you want to use for the tracer.")

        compsep_run.setdefault("reduce_mcilc_bias", True)
        compsep_run.setdefault("n_patches", 50)
        compsep_run.setdefault("save_patches", False)
                                    
    compsep_run.setdefault("save_weights", save_products)
        
    return compsep_run
    

def _check_fields(config: Configs, data: SimpleNamespace) -> Configs:
    """
    Validate and infer field_in and field_out from config and data.

    Parameters
    ----------
        config : Configs
            The configuration object.

        data : Namespace
            The input data with at least the 'total' attribute.

    Returns
    -------
        Configs
            Updated configuration object.

    Raises
    ------
        ValueError
            If fields or data are invalid.
    """
    if not config.field_in:
        raise ValueError("field_in must be provided.")

    valid_fields = {
        "maps": ["T", "E", "B", "QU", "EB", "TQU", "TEB"],
        "alms": ["T", "E", "B", "EB", "TEB"]
    }

    if config.field_in not in valid_fields.get(config.data_type, []):
        raise ValueError(f"Invalid field_in for {config.data_type}. It must be one of {valid_fields[config.data_type]}.")

    if data.total.ndim == 2:
        if config.field_in not in ["T", "E", "B"]:
            raise ValueError("Invalid value for field_in. It must be 'T', 'E', or 'B'.")
        if not config.field_out:
            config.field_out = config.field_in
        elif config.field_in == "T" and config.field_out != "T":
            raise ValueError("field_out must be 'T' when field_in is 'T'.")
        elif config.field_in in ["E", "B"]:
            if config.field_out not in [config.field_in, "QU", f"QU_{config.field_in}"]:
                raise ValueError(f"Invalid value for field_out given the provided field_in. It must be 'QU', '{config.field_in}' or 'QU_{config.field_in}'.")
            if config.field_out == "QU":
                config.field_out = "QU_" + config.field_in
    elif data.total.ndim==3:
        if data.total.shape[1]==2:
            if config.field_in not in ["QU", "EB"]:
                raise ValueError("field_in must be 'QU' or 'EB' for 2-field data.")
        elif data.total.shape[1]==3:
            if config.field_in not in ["TQU", "TEB"]:
                raise ValueError("field_in must be 'TQU' or 'TEB' for 3-field data.")

        if config.field_out:
            valid_outs = {
                2: ["QU", "EB", "E", "B", "QU_E", "QU_B"],
                3: ["T", "E", "B", "EB", "QU", "QU_E", "QU_B", "TQU", "TEB"]
            }
            if config.field_out not in valid_outs.get(data.total.shape[1], []):
                raise ValueError(f"Invalid field_out. Must be one of {valid_outs.get(data.total.shape[1])}.")
        else:
            config.field_out = config.field_in

    return config

def _get_field_in_cs(field_in: str, field_out: str) -> str:
    """
    Infer the component separation fields based on inputs with multiple fields.

    Parameters
    ----------
        field_in : str
            The input fields (e.g., "TQU", "TEB").
        field_out : str
            The desired output field from component separation.

    Returns
    -------
        str
            The appropriate field_in_cs for component separation.
    """
    if field_in == "TEB":
        if field_out in ["T", "E", "B", "EB"]:
            return field_out
        elif field_out == "QU":
            return "EB"
        elif field_out in ["QU_E", "QU_B"]:
            return field_out[-1]
        else:
            return field_in
    elif field_in == "TQU":
        if field_out in ["T", "QU"]:
            return field_out
        elif field_out in ["E", "B", "EB", "QU_E", "QU_B"]:
            return "QU"
        else:
            return field_in
    elif field_in == "EB":
        if field_out in ["E", "B"]: 
            return field_out
        elif field_out in ["QU_E", "QU_B"]:
            return field_out[-1]
        else:
            return field_in
    elif field_in == "QU":
        return field_in

def _check_data_and_config(config: Configs, data: SimpleNamespace) -> Configs:
    """
    Validate input data and update config with resolution info.

    Parameters
    ----------
        config : Configs
            The configuration object.
        data : Namespace
            The input data.

    Returns
    -------
        Configs
            Updated config object.

    Raises
    ------
        ValueError
            If data or configuration is invalid.
    """

    if config.data_type not in ["maps", "alms"]:
        raise ValueError("data_type must be 'maps' or 'alms'.")

    if not hasattr(data, "total"):
        raise ValueError("data must have a 'total' attribute.")

    allowed_attributes = ["total", "noise", "cmb", "nuisance", "fgds", "dust", "synch", "ame", "co", "freefree", "cib", "tsz", "ksz", "radio_galaxies"]
    if not all(attr in allowed_attributes for attr in vars(data).keys()):
        raise ValueError(f"The provided data have invalid attributes. Allowed attributes are: {allowed_attributes}.")

    if len(vars(data)) > 1:
        attr_shape = next(iter(vars(data).values())).shape 
        if not all(attr.shape == attr_shape for attr in vars(data).values()):
            raise ValueError("All data components must have the same shape.")    

    if config.data_type == "maps":
        try:
            nside_in = hp.npix2nside(data.total.shape[-1])
        except:
            raise ValueError("Invalid number of pixels in data.")
        if config.lmax >= 3*nside_in:
            raise ValueError("lmax is too high for the provided nside. It must be smaller than 3*nside.")
        if not hasattr(config, "nside_in"):
            config.nside_in = nside_in
    elif config.data_type == "alms":
        try:
            lmax_in =  hp.Alm.getlmax(data.total.shape[-1])
        except:
            raise ValueError("Invalid number of alms in data.")
        if config.lmax > lmax_in:
            print("Required lmax is larger than that of provided alms. lmax updated to the maximum multipole of the provided alms.")
            config.lmax = lmax_in
        config.lmax_in = lmax_in

    if data.total.ndim == 3:
        if data.total.shape[1] > 3:
            raise ValueError("The provided data have wrong number of fields. It must be 1, 2 or 3.")
    if data.total.ndim > 3:
        raise ValueError("The provided data have wrong number of dimensions. It must be 2 or 3.")
    return config

def _load_outputs(path: str, components:list, field_out: str, 
    nside: int, lmax: int, fwhm_out: float, nsim: Optional[str] = None) -> np.ndarray:
    """
    Load component separation outputs from specified path.

    Parameters
    ----------
        path : str
            Path to the directory containing the component separation outputs.
        components : list or str
            List of component names or a single component name to load.
        field_out : str
            The field(s) returned and stored by the component separation run of interest.
        nside : int
            HEALPix resolution associated to the component separation outputs.
        lmax : int
            Maximum multipole for the component separation outputs.
        fwhm_out : float
            Full width at half maximum of the output maps in arcminutes.
        nsim : int or str, optional
            Simulation index of the compsep outputs. If None, it will look for files without nsim label. Default: None.

    Returns
    -------
        np.ndarray
            Loaded HEALPix map(s) for the specified components.
        
    """

    loaded_outputs = []

    if isinstance(components, str):
        components = [components]
    if not isinstance(components, list):
        raise ValueError("components must be a string or a list of strings.")

    for component in components:
        component_name = component.split('/')[0] if '/' in component else component
        filename = os.path.join(
            path,
            f"{component}/{field_out}_{component_name}_{fwhm_out}acm_ns{nside}_lmax{lmax}"
        )
        loaded_outputs.append(_load_outputs_(filename, field_out, nsim=nsim))

    return np.array(loaded_outputs)


def _load_outputs_(filename: str, fields: str, nsim: Optional[str] = None) -> np.ndarray:
    """
    Load HEALPix map outputs based on field type and optional simulation number.

    Parameters
    ----------
        filename : str
            Base filename (without extension).
        fields : str
            The field(s) to read (e.g., 'TQU', 'EB').
        nsim : int or None
            Simulation index to append to filename.

    Returns
    -------
        np.ndarray
            Loaded HEALPix map(s).

    Raises
    ------
        ValueError
            If the fields parameter is invalid.
    """

    nsim = _format_nsim(nsim)

    if nsim is not None:
        filename += f"_{nsim}.fits"
        
    if fields in ["TEB", "TQU"]:
        return hp.read_map(filename, field=[0,1,2])
    elif fields in ["EB", "QU", "QU_E", "QU_B"]:
        return hp.read_map(filename, field=[0,1])
    elif fields in ["T", "E", "B"]:
        return hp.read_map(filename, field=[0])
    else:
        raise ValueError("Invalid value for fields. It must be 'T', 'E', 'B', 'EB', 'QU', 'TQU', 'TEB', 'QU_E', or 'QU_B'.")


def _combine_products(config: Configs, nsim=None):
    """
    Combine fields from multiple runs into a single output file.

    Parameters
    ----------
        config : Configs
            Configuration object with settings for combining products. It should include:
            - combine_outputs (list): List of dictionaries with settings for each combination run. Each dictionary should contain:
                - fields_in (list): List of fields associated to the compsep products to be combined. The elements of the list should be different one to another.
                - paths_fields_in (list): List of paths to the directories containing the compsep products to be combined.
                - path_out (str): Path to the directory where the combined products will be saved.
                - fields_out (str): Fields of the combined products. Default: ''.join(fields_in).
                - components (list): List of components returned by component separation runs to be combined. Default: ['output_total'].
    
        nsim : int or str, optional
            Simulation number to be used for loading the component separation products and saving the outputs. 
            If None, the outputs will be saved without label on simulation number.
            Default: None.

    Returns
    -------
        SimpleNamespace or None
            If `config.return_compsep_products` is True, returns outputs. Otherwise returns None.
    
    Raises
    ------
        ValueError
            If the configuration is invalid or required fields are missing.
    """

    nsim = _format_nsim(nsim)

    mask_obs, mask_cov = get_masks_for_compsep(config.mask_observations, config.mask_covariance, config.nside)
    
    for combine_run in config.combine_outputs:
        if ("fields_in" not in combine_run) or ("paths_fields_in" not in combine_run) or ("path_out" not in combine_run):
            raise ValueError("'fields_in', 'paths_fields' and 'path_out' must be provided in the combine_outputs dictionary.")

        if not isinstance(combine_run["fields_in"], list) or not isinstance(combine_run["paths_fields_in"], list):
            raise ValueError("'fields_in' and 'paths_fields_in' must be lists of strings.")

        if "fields_out" not in combine_run:
            combine_run["fields_out"] = "".join(combine_run["fields_in"])

        if "components" not in combine_run:
            combine_run["components"] = ["output_total"]

        if config.return_compsep_products:
            outputs_recomb = SimpleNamespace()

        for component in combine_run["components"]:
            if config.save_compsep_products:
                os.makedirs(os.path.join(config.path_outputs, combine_run["path_out"], component), exist_ok=True)
        
            outputs = None
            component_name = component.split('/')[0] if '/' in component else component

            for field_in, path in zip(combine_run["fields_in"], combine_run["paths_fields_in"]):
                path_in = os.path.join(config.path_outputs, path)
                filename = os.path.join(path_in, f"{component}/{field_in}_{component_name}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}")
                if outputs is None:
                    outputs = _load_outputs_(filename, field_in, nsim=nsim)
                    if outputs.ndim == 1:
                        outputs = outputs[np.newaxis, :]
                else:
                    outputs_ = _load_outputs_(filename,field_in,nsim=nsim)
                    if outputs_.ndim == 1:
                        outputs_ = outputs_[np.newaxis, :]
                    outputs = np.concatenate([outputs,outputs_], axis=0)
                    del outputs_

            if outputs.ndim == 2 and outputs.shape[0] == 1:
                outputs = outputs[0]

            if combine_run["fields_out"] != "".join(combine_run["fields_in"]):
                if combine_run["fields_in"] in [["T", "E", "B"], ["T", "EB"]]:
                    if combine_run["fields_out"] == "TQU":
                        outputs = _EB_to_QU(outputs, config.lmax)
                    else:
                        raise ValueError(f"'field_out' can be {''.join(combine_run['fields_in'])} or TQU for provided 'field_in'")
                elif combine_run["fields_in"] in [["T", "E"], ["T", "B"]]:
                    if combine_run["fields_out"][:3] == "TQU":
                        if combine_run["fields_out"] == "TQU":
                            combine_run["fields_out"] = f"TQU_{combine_run['fields_in'][-1]}"
                        if "E" in combine_run["fields_in"]:
                            outputs = np.concatenate([(outputs[0])[np.newaxis, :], _E_to_QU(outputs[1], config.lmax)], axis=0)
                        elif "B" in combine_run["fields_in"]:
                            outputs = np.concatenate([(outputs[0])[np.newaxis, :], _B_to_QU(outputs[1], config.lmax)], axis=0)
                    else:
                        raise ValueError(f"'field_out' can be {''.join(combine_run['fields_in'])}, TQU or TQU_{combine_run['fields_in'][-1]} for provided 'field_in'")
                elif combine_run["fields_in"] in [["E", "B"], ["E"], ["B"]]:
                    if combine_run["fields_out"][:2] != "QU":
                        raise ValueError(f"Invalid 'fields_out' for provided 'field_in'")                        
                    if outputs.ndim == 2:
                        combine_run["fields_out"] = "QU"
                        outputs = _EB_to_QU(outputs, config.lmax)
                    elif outputs.ndim==1:
                        combine_run["fields_out"] = f"QU_{combine_run['fields_in'][0]}"
                        if combine_run['fields_in'][0] == "E":
                            outputs = _E_to_QU(outputs, config.lmax)
                        elif combine_run['fields_in'][0] == "B":
                            outputs = _B_to_QU(outputs, config.lmax)
                elif combine_run["fields_in"] in [["T", "QU"], ["QU"]]:
                    if (combine_run["fields_in"]==["T", "QU"] and combine_run["fields_out"]=="TEB") or (combine_run["fields_in"]==["QU"] and combine_run["fields_out"]=="EB"):
                        outputs = _QU_to_EB(outputs, config.lmax)
                    else:
                        raise ValueError(f"Wrong 'field_out' for provided 'field_in'")
                else:
                    raise ValueError(f"Invalid 'field_in' or 'field_out'")
            
            if mask_cov is not None:
                if outputs.ndim == 1:
                    outputs[mask_cov == 0.] = 0.
                elif outputs.ndim == 2:
                    outputs[:, mask_cov == 0.] = 0.
                
            if config.save_compsep_products:
                filename_out = os.path.join(config.path_outputs, combine_run["path_out"], f"{component}/{combine_run['fields_out']}_{component_name}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}")
                if nsim is not None:
                    filename_out += f"_{nsim}"
                filename_out += f".fits"
                hp.write_map(filename_out, outputs, overwrite=True)

            if config.return_compsep_products:
                setattr(outputs_recomb, component_name, np.array(outputs))

    if config.return_compsep_products:
        return outputs_recomb


__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]
                    

                    


        
