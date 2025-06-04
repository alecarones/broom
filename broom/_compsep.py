import time
import warnings
from functools import partialmethod
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm
import healpy as hp
from types import SimpleNamespace

from .configurations import Configs
from .routines import _alms_from_data, merge_dicts, _map2alm_kwargs, _slice_data
from ._gilcs import gilc, get_gnilc_maps, fgd_diagnostic
from ._ilcs import ilc#, cilc, pilc
from ._pilcs import pilc
from ._gpilcs import gpilc, fgd_P_diagnostic
from ._templates import get_residuals_template

def get_data_and_compsep(config: Configs, foregrounds, nsim=None):
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

def component_separation(config: Configs, data, nsim = None, **kwargs):  # shape (N_sims, lmax - 1)
    r"""
    Runs component separation methods on the provided data.

    Parameters
    ----------

    config : Configs
        Configuration object containing the settings for component separation.
        In particular, it should contain the following attributes:
        - `data_type`: Type of data, either "maps" or "alms".
        - `field_in`: Field associated to the provided data.
        - `field_out`: Fields of the outputs from component separation. Default: `field_in`.
        - `lmax`: Maximum multipole for the component separation.
        - `nside`: HEALPix resolution of the outputs.
        - 'fwhm_out': Full width at half maximum of the output maps in arcminutes.
        - `compsep`: List of dictionaries containing the configuration for each component separation method to be run.
        - `mask_path`: Path to the HEALPix mask fits file, if any. Default: None.
        - `mask_type`: Type of mask, either "mask_for_compsep" (only used to exclude regions in covariance computation) or "observed_patch". 
                    Default: "mask_for_compsep".
        - 'leakage_correction': Whether to apply EB-leakage correction in input data if mask_type is "observed_patch". 
                    Default: None.
        - `bring_to_common_resolution`: Whether to bring the data to a common resolution. If False, the data will be used as is. Default: True.
        - `pixel_window_in`: Whether to pixel window is included in the input data. Default: False.
        - 'pixel_window_out': Whether to include pixel window in the output data. Default: False.
        - `save_compsep_products`: Whether to save the outputs of component separation. Default: True.
        - `return_compsep_products`: Whether to return the outputs of component separation. Default: False.
        - 'path_outputs': Path to the directory where the outputs will be saved if 'save_compsep_products' is True. 
                    Default: Working directory + "/outputs/".
        - `verbose`: Whether to print information about the component separation process. Default: False.
    
    data : SimpleNamespace
        Data object containing the input data for component separation. It should have the following attributes:
        - `total`: Total map or alms to be used for component separation.
        - Other optional attributes such as `noise`, `cmb`, `fgds`, etc.

    nsim : Optional[int or str], optional
        Simulation number to be used for saving the outputs. If None, the outputs will be saved without label on simulation number.
        Default: None.

    """
    
    if nsim is not None:
        if not isinstance(nsim, (int, str)):
            raise ValueError("Invalid value for nsim. It must be an integer or a string.")
        if isinstance(nsim, int):
            nsim = str(nsim).zfill(5)
    else:
        if config.verbose:
            print(f"Simulation number not provided. If 'save_compsep_products' is set to True, the outputs will be saved without label on simulation number.")
            
    kwargs = _map2alm_kwargs(**kwargs)
    config = _check_data_and_config(config, data)
    config = _check_fields(config, data)

    if (data.total.ndim > 2) and (config.field_out != config.field_in):
        config.field_in_cs = _get_field_in_cs(config.field_in, config.field_out)
        data = _slice_data(data, config.field_in, config.field_in_cs)
        print(data.total.shape)
    else:
        config.field_in_cs = config.field_in

    if config.verbose:
        if nsim is None:
            print(f"Computing required input alms for component separation." if config.data_type=="maps" else f"Pre-processing input alms for component separation.")        
        else:
            print(f"Computing required input alms for component separation for simulation {nsim}." if config.data_type=="maps" else f"Pre-processing input alms for component separation for simulation {nsim}.")
    
    if config.mask_path is not None:
        if not hasattr(config, "mask_type"):
            config.mask_type = "mask_for_compsep"
        if config.mask_type not in ["mask_for_compsep", "observed_patch"]:
            raise ValueError("Invalid value for 'mask_type'. It can be either 'mask_for_compsep' or 'observed_patch'.")
        else:
            if config.verbose:
                print(f"Provided mask is used as" + " observed patch" if config.mask_type == "observed_patch" else " mask for component separation.")
        if not isinstance(config.mask_path, str):
            raise ValueError("Invalid mask_path in config. It must be a string full path to a HEALPix mask fits file.")
        mask = hp.read_map(config.mask_path, field=0)

        if (config.mask_type == "observed_patch") and (config.data_type == "maps"):
            input_alms = _alms_from_data(config, data, config.field_in_cs, mask_in=_preprocess_mask(mask, config.nside_in), data_type=config.data_type, bring_to_common_resolution=config.bring_to_common_resolution, pixel_window_in=config.pixel_window_in, **kwargs)
        else:
            input_alms = _alms_from_data(config, data, config.field_in_cs, data_type=config.data_type, bring_to_common_resolution=config.bring_to_common_resolution, pixel_window_in=config.pixel_window_in, **kwargs)
    else:
        mask = None
        input_alms = _alms_from_data(config, data, config.field_in_cs, data_type=config.data_type, bring_to_common_resolution=config.bring_to_common_resolution, pixel_window_in=config.pixel_window_in, **kwargs)

    methods_map = {
    'ilc': ilc,
    'gilc': gilc,
    'cilc': ilc,
    'c_ilc': ilc,
    'mc_ilc': ilc,
    'mcilc': ilc,
    'pilc': pilc,
    'cpilc': pilc,
    'c_pilc': pilc,
    'gpilc': gpilc,
    'fgd_diagnostic': fgd_diagnostic,
    'fgd_P_diagnostic': fgd_P_diagnostic
    }
    #}

    if config.return_compsep_products:
        outputs = SimpleNamespace()
        if any(compsep_run["method"] not in ["fgd_diagnostic", "fgd_P_diagnostic"] for compsep_run in config.compsep):
        # Initialize attributes of `outputs` with empty lists
            for attr in vars(data):
                setattr(outputs, attr, [])
        
    if config.verbose:
        print(f"Running component separation for simulation {nsim}." if nsim is not None else f"Running component separation.")

    for compsep_run in config.compsep:
        compsep_run = _standardize_compsep_config(compsep_run, save_products=config.save_compsep_products)
        compsep_run["nsim"] = nsim

        if mask is not None:
            compsep_run["mask"] = _preprocess_mask(mask, config.nside)

        if config.verbose:
            print(f"Running {compsep_run['method']} in {compsep_run['domain']} domain for simulation {nsim}." if nsim is not None else f"Running {compsep_run['method']} in {compsep_run['domain']} domain.")
        
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

        if "mask" in compsep_run:
            del compsep_run['mask']
        del compsep_run["nsim"]

    if config.return_compsep_products:
        for attr in vars(outputs).keys():
            setattr(outputs, attr, np.array(getattr(outputs, attr)))
        return outputs

def estimate_residuals(config: Configs, nsim = None, **kwargs):  # shape (N_sims, lmax - 1)
    r"""

    """

    if nsim is not None:
        if not isinstance(nsim, (int, str)):
            raise ValueError("Invalid value for nsim. It must be an integer or a string.")
        if isinstance(nsim, int):
            nsim = str(nsim).zfill(5)
    else:
        if config.verbose:
            print(f"Simulation number not provided. If 'save_compsep_products' is set to True, the outputs will be saved without label on simulation number.")        

    kwargs = _map2alm_kwargs(**kwargs)

    if config.mask_path is not None:
        if not hasattr(config, "mask_type"):
            config.mask_type = "mask_for_compsep"
        if config.mask_type not in ["mask_for_compsep", "observed_patch"]:
            raise ValueError("Invalid value for 'mask_type'. It can be either 'mask_for_compsep' or 'observed_patch'.")
        else:
            if config.verbose:
                print(f"Provided mask is used as" + "observed patch" if config.mask_type == "observed_patch" else "mask for component separation.")
        mask = hp.read_map(config.mask_path, field=0)
    else:
        mask = None
    
    if config.verbose:
        print(f"Running foregrounds residuals estimation for simulation {nsim}." if nsim is not None else f"Running component separation.")

    if config.return_compsep_products:
        outputs = SimpleNamespace()

    for compsep_run in config.compsep_residuals:
        if "field_in" not in compsep_run:
            compsep_run["field_in"] = config.field_out
            delete_field_in = True
        else:
            delete_field_in = False
        compsep_run["nsim"] = nsim

        if not "adapt_nside" in compsep_run:
            compsep_run["adapt_nside"] = False

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

        if mask is not None:
            if (config.mask_type == "observed_patch"):
                input_alms = _alms_from_data(config, data, compsep_run["field_in_cs"], mask_in=_preprocess_mask(mask, config.nside_in), data_type="maps", bring_to_common_resolution=False, pixel_window_in=config.pixel_window_out, **kwargs)
            else:
                input_alms = _alms_from_data(config, data, compsep_run["field_in_cs"], data_type="maps", bring_to_common_resolution=False, pixel_window_in=config.pixel_window_out, **kwargs)
        else:
            input_alms = _alms_from_data(config, data, compsep_run["field_in_cs"], data_type="maps", bring_to_common_resolution=False, pixel_window_in=config.pixel_window_out, **kwargs)

        if mask is not None:
            compsep_run["mask"] = _preprocess_mask(mask, config.nside)

        if config.return_compsep_products:
            prod = get_residuals_template(config, input_alms, compsep_run, **kwargs)
            for attr in vars(prod).keys():
                getattr(outputs, attr).append(getattr(prod, attr))
        else:
            get_residuals_template(config, input_alms, compsep_run, **kwargs)

        if "mask" in compsep_run:
            del compsep_run['mask']
        del compsep_run["nsim"]
        if delete_field_in:
            del compsep_run["field_in"]

    if config.return_compsep_products:
        for attr in vars(outputs).keys():
            setattr(outputs, attr, np.array(getattr(outputs, attr)))
        return outputs

def _standardize_compsep_config(compsep_run, save_products=True):
    if compsep_run["domain"] == "pixel":
        compsep_run["b_squared"] = False 
        compsep_run["adapt_nside"] = False 
    if compsep_run["domain"] == "needlet":
        # Update the original dictionary with the merged dictionary
        if "needlet_config" not in compsep_run:
            raise ValueError("needlet_config must be provided if compsep domain is needlet.")
        compsep_run['needlet_config'] = merge_dicts(compsep_run['needlet_config'])
        if "b_squared" not in compsep_run:
            compsep_run["b_squared"] = False
        if "adapt_nside" not in compsep_run:
            compsep_run["adapt_nside"] = False    
        if not "save_needlets" in compsep_run["needlet_config"]:
            if save_products:
                compsep_run['needlet_config']['save_needlets'] = True
            else:
                compsep_run['needlet_config']['save_needlets'] = False 

    if compsep_run["method"] != "mcilc":
        if "ilc_bias" not in compsep_run:
            compsep_run["ilc_bias"] = 0.001

    if compsep_run["domain"] == "pixel" or compsep_run["domain"] == "needlet":
        if ("reduce_ilc_bias" not in compsep_run) and (compsep_run["method"] != "mcilc"):
            compsep_run["reduce_ilc_bias"] = False
        if compsep_run["method"] in ["mcilc", "mc_ilc"]:
            if "reduce_mcilc_bias" not in compsep_run:
                compsep_run["reduce_mcilc_bias"] = True
    
    if compsep_run["method"] in ["c_ilc","c_pilc","mc_ilc"] and compsep_run["domain"] != "needlet":
        raise ValueError("The methods 'c_ilc', 'c_pilc' and 'mc_ilc' can only be used in the needlet domain.")

    if compsep_run["method"] in ["c_ilc", "c_pilc", "mc_ilc"]:
        if "special_nls" not in compsep_run:
            raise ValueError("special_nls must be provided for methods 'c_ilc', 'c_pilc' and 'mc_ilc'.")      
        if not isinstance(compsep_run["special_nls"], list):
            raise ValueError("special_nls must be a list of integers.")
        
    if compsep_run["method"] in ["cilc", "c_ilc", "cpilc", "c_pilc"]:
        if "constraints" not in compsep_run:
            raise ValueError("A dictionary of constraints must be provided in the compsep_run dictionary for methods 'cilc', 'c_ilc', 'cpilc', 'c_pilc'.")
        else:
            compsep_run['constraints'] = merge_dicts(compsep_run['constraints'])
    
    if compsep_run["method"] in ["mcilc", "mc_ilc"]:
        if "mc_type" not in compsep_run:
            compsep_run["mc_type"] = "cea_real"
        if compsep_run["mc_type"] not in ["cea_ideal","cea_real","rp_ideal","rp_real"]:
            raise ValueError("Invalid value for mc_type. It must be 'cea_ideal', 'cea_real', 'rp_ideal' or 'rp_real'.")

        if "real" in compsep_run["mc_type"]:
            if "path_tracers" not in compsep_run:
                raise ValueError("Path to tracers ('path_tracers') must be provided for methods 'mcilc' and 'mc_ilc'.")
            if not isinstance(compsep_run["path_tracers"], str):
                raise ValueError("'path_tracers' must be a string.")
            compsep_run["path_tracers"] = compsep_run["path_tracers"] if compsep_run["path_tracers"].endswith('/') else compsep_run["path_tracers"] + '/'

        if "channels_tracers" not in compsep_run:
            raise ValueError("channels_tracers must be provided for methods 'mcilc' and 'mc_ilc'. It must be a list of two integers corresponding to the indices of the channels you want to use for the tracer.")
        else:
            if (not isinstance(compsep_run["channels_tracers"], list)) or (len(compsep_run["channels_tracers"]) != 2):
                raise ValueError("channels_tracers must be a list of two integers corresponding to the indices of the channels you want to use for the tracer.")

        if "reduce_mcilc_bias" not in compsep_run:
            compsep_run["reduce_mcilc_bias"] = True

        if "n_patches" not in compsep_run:
            compsep_run["n_patches"] = 50
                                    
    if "save_weights" not in compsep_run:
        compsep_run["save_weights"] = False
        
    return compsep_run
    
def _preprocess_mask(mask, nside_out):
    if isinstance(mask, np.ndarray):
        try:
            nside_mask = hp.get_nside(mask)
            if not is_binary_mask(mask):
                print("Provided mask is not binary. Mask is assumed to be a normalized hits map.")
            if nside_mask < nside_out:
                print("Provided mask has lower HEALPix resolution than that required for outputs. Mask will be upgraded to the output resolution.")
                mask = _upgrade_mask(mask, nside_out)
            elif nside_mask > nside_out:
                print("Provided mask has higher HEALPix resolution than that required for outputs. Mask will be downgraded to the output resolution.")
                mask = _downgrade_mask(mask, nside_out, threshold=0.5)
        except:
            raise ValueError("Invalid mask. It must be a valid HEALPix mask.")
        return mask
    else:
        raise ValueError("Invalid mask. It must be a numpy array.")

def _upgrade_mask(mask, nside_out):
    if is_binary_mask(mask):
        return hp.ud_grade(mask, nside_out)
    else:
        return hp.ud_grade(mask, nside_out, power=-2)

def _downgrade_mask(mask, nside_out, threshold=0.5):
    if is_binary_mask(mask):
        mask = hp.ud_grade(mask, nside_out)
        mask[mask <= threshold] = 0.
        mask[mask > threshold] = 1.
        return mask
    else:
        return hp.ud_grade(mask, nside_out, power=-2)

def is_binary_mask(mask):
    return np.all(np.isin(mask, [0., 1.]))

def _check_fields(config: Configs, data):
    if not config.field_in:
        raise ValueError("field_in must be provided.")
    if config.data_type == "maps":
        if config.field_in not in ["T", "E", "B", "QU", "EB", "TQU", "TEB"]:
            raise ValueError("Invalid value for field_in. It must be 'T', 'E', 'B', 'QU', 'EB', 'TQU', or 'TEB'.")
    elif config.data_type == "alms":
        if config.field_in not in ["T", "E", "B", "EB", "TEB"]:
            raise ValueError("Invalid value for field_in. It must be 'T', 'E', 'B', 'EB', or 'TEB'.")
    if data.total.ndim == 2:
        if config.field_in not in ["T", "E", "B"]:
            raise ValueError("Invalid value for field_in. It must be 'T', 'E', or 'B'.")
        if config.field_out:
            if config.field_in == "T" and config.field_out != "T":
                raise ValueError("Invalid value for field_out. It must be 'T' for the provided field_in.")
            elif config.field_in == "E" or config.field_in == "B":
                if config.field_out not in [config.field_in, "QU", f"QU_{config.field_in}"]:
                    raise ValueError(f"Invalid value for field_out given the provided field_in. It must be 'QU', '{config.field_in}' or 'QU_{config.field_in}'.")
                if config.field_out == "QU":
                    config.field_out = "QU_" + config.field_in
        else:
            config.field_out = config.field_in
    elif data.total.ndim==3:
        if data.total.shape[1]==2:
            if config.field_in not in ["QU", "EB"]:
                raise ValueError("Invalid value for field_in. It must be 'QU' or 'EB' for the provided data.")
        if data.total.shape[1]==3:
            if not config.field_in in ["TQU", "TEB"]:
                raise ValueError("Invalid value for field_in. It must be 'TQU' or 'TEB' for the provided data.")
        if config.field_out:
            if data.total.shape[1]==2:
                if config.field_out not in ["QU", "EB", "E", "B", "QU_E", "QU_B"]:
                    raise ValueError("Invalid value for field_out. It must be 'QU', 'EB', 'E' or 'B' for the provided data.")
            elif data.total.shape[1]==3:
                if config.field_out not in ["T", "E", "B", "EB", "QU", "QU_E", "QU_B", "TQU", "TEB"]:
                    raise ValueError("Invalid value for field_out. It must be 'T', 'E', 'B', 'EB', 'QU', 'QU_E', 'QU_B', 'TQU' or 'TEB' for the provided data.")
        else:
            config.field_out = config.field_in
    return config

def _get_field_in_cs(field_in, field_out):
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

def _check_data_and_config(config: Configs, data):
    if config.data_type not in ["maps", "alms"]:
        raise ValueError("Invalid value for data_type. It must be 'maps' or 'alms'.")

    if not hasattr(data, "total"):
        raise ValueError("The provided data do not have the 'total' attribute.")
    
    allowed_attributes = ["total", "noise", "cmb", "fgds", "dust", "synch", "ame", "co", "freefree", "cib", "tsz", "ksz", "radio_galaxies"]
    if not all(attr in allowed_attributes for attr in vars(data).keys()):
        raise ValueError(f"The provided data have invalid attributes. Allowed attributes are: {allowed_attributes}.")

    if len(vars(data)) > 1:
        attr_shape = next(iter(vars(data).values())).shape 
        if not all(attr.shape == attr_shape for attr in vars(data).values()):
            raise ValueError("The provided data have different shapes for different attributes.")
    
    if config.data_type == "maps":
        try:
            nside_in = hp.npix2nside(data.total.shape[-1])
        except:
            raise ValueError("The provided data have wrong number of pixels.")
        if config.lmax >= 3*nside_in:
            raise ValueError("lmax is too high for the provided nside. It must be smaller than 3*nside.")
        config.nside_in = nside_in
    elif config.data_type == "alms":
        try:
            lmax_in =  hp.Alm.getlmax(data.total.shape[-1])
        except:
            raise ValueError("The provided data have wrong number of alms.")
        if config.lmax > lmax_in:
            print("Required lmax is larger than that of provided alms. lmax updated to the maximum multipole of the provided alms.")
            config.lmax = lmax_in
        config.lmax_in = lmax_in
#    if config.experiment:
#        if data.shape[0] != len(config.instrument.frequency):
#            raise ValueError("The number of provided data does not match the number of frequencies in the experiment database.")
    if data.total.ndim == 3:
        if data.total.shape[1] > 3:
            raise ValueError("The provided data have wrong number of fields. It must be 1, 2 or 3.")
    if data.total.ndim > 3:
        raise ValueError("The provided data have wrong number of dimensions. It must be 2 or 3.")
    return config

def _load_outputs(filename,fields,nsim=None):
    if nsim is not None:
        filename = filename + f"_{nsim}.fits"
    if fields == "TEB" or fields == "TQU":
        return hp.read_map(filename, field=[0,1,2])
    elif fields == "EB" or fields[:2] == "QU":
        return hp.read_map(filename, field=[0,1])
    elif fields == "T" or fields == "E" or fields == "B":
        return hp.read_map(filename, field=[0])
    else:
        raise ValueError("Invalid value for fields. It must be 'T', 'E', 'B', 'EB', 'QU', 'TQU', 'TEB', 'QU_E', or 'QU_B'.")


#def _combine_products(config: Configs, nsim=None):
#    if nsim is not None:
#        if not isinstance(nsim, (int, str)):
#            raise ValueError("Invalid value for nsim. It must be an integer or a string.")
#        if isinstance(nsim, int):
#            nsim = str(nsim).zfill(5)
#    
#    if ("fields_in" not in config.combine_products) or ("paths_fields" not in config.combine_products) or ("path_out" not in config.combine_products):
#        raise ValueError("'fields_in', 'paths_fields' and 'path_out' must be provided in the combine_products dictionary.")

#    if "fields_out" not in config.combine_products:
#        config.combine_products["fields_out"] = "".join(config.combine_products["fields_in"])

#    if "components" not in config.combine_products:
#        config.combine_products["components"] = ["output_total"]
    
#    outputs = []
#    for field_in in config.combine_products["fields_in"]:
#        outputs_ = []
#        for component in components:
#            filename = config.path_outputs + f"/{config.field_in}_{component}_{config.fwhm_out}acm_ns{config.nside}"
#            outputs.append(_load_outputs(filename,config.field_in,nsim=nsim))
#    outputs = np.array(outputs)

#    if config.combine_products["fields_in"] in [["T", "E", "B"], ["T", "EB"], ["E", "B"], ["EB"]]:
        
