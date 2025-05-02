import time
import warnings
from functools import partialmethod
from typing import Optional, Tuple

import numpy as np
#from mpi4py import MPI
from tqdm import tqdm
import healpy as hp
from types import SimpleNamespace

from .configurations import Configs
from .routines import _alms_from_data, merge_dicts
from ._gilcs import gilc, get_gnilc_maps
from ._ilcs import ilc#, cilc, pilc
from ._pilcs import pilc
from ._gpilcs import gpilc
from ._fres import get_residuals_template

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

def component_separation(config: Configs, data, nsim = None, mask = None):  # shape (N_sims, lmax - 1)
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

    config = _check_data_and_config(config, data)
    config = _check_fields(config, data)
    if config.verbose:
        if nsim is None:
            print(f"Computing required input alms for component separation." if config.data_type=="maps" else f"Pre-processing input alms for component separation.")        
        else:
            print(f"Computing required input alms for component separation for simulation {nsim}." if config.data_type=="maps" else f"Pre-processing input alms for component separation for simulation {nsim}.")
    
    if mask is not None:
        if not hasattr(config, "mask_type"):
            remove_mask_type = True
            config.mask_type = "mask_for_compsep"
        else:
            remove_mask_type = False
            
        if config.mask_type not in ["mask_for_compsep", "observed_patch"]:
            raise ValueError("Invalid value for 'mask_type'. It can be either 'mask_for_compsep' or 'observed_patch'.")
        else:
            if config.verbose:
                print(f"Provided mask is used as" + "observed patch" if config.mask_type == "observed_patch" else "mask for component separation.")

        if (config.mask_type == "observed_patch") and (config.data_type == "maps"):
            mask_in = _preprocess_mask(mask, config.nside_in)
            input_alms = _alms_from_data(config, data, mask_in=mask_in, data_type=config.data_type, bring_to_common_resolution=config.bring_to_common_resolution, pixel_window_in=config.pixel_window_in)
        else:
            input_alms = _alms_from_data(config, data, data_type=config.data_type, bring_to_common_resolution=config.bring_to_common_resolution, pixel_window_in=config.pixel_window_in)

#        if config.mask_type == "mask_for_compsep":
#            if config.leakage_correction is not None:
#                config.leakage_correction = None
    else:
        input_alms = _alms_from_data(config, data, data_type=config.data_type, bring_to_common_resolution=config.bring_to_common_resolution, pixel_window_in=config.pixel_window_in)

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
    }
    #}

    if config.return_compsep_products:
        outputs = SimpleNamespace()
        for attr in vars(data):
            setattr(outputs, attr, [])
        
    if config.verbose:
        print(f"Running component separation for simulation {nsim}." if nsim is not None else f"Running component separation.")

    for compsep_run in config.compsep_runs:
        compsep_run = _standardize_compsep_config(compsep_run, save_products=config.save_compsep_products)

        if mask is not None:
            compsep_run["mask"] = _preprocess_mask(mask, config.nside)

        if config.verbose:
            print(f"Running {compsep_run['method']} in {compsep_run['domain']} domain for simulation {nsim}." if nsim is not None else f"Running {compsep_run['method']} in {compsep_run['domain']} domain.")
        if config.return_compsep_products:
            prod = methods_map[compsep_run["method"]](config, input_alms, compsep_run, nsim=nsim)
            for attr in vars(prod).keys():
                getattr(outputs, attr).append(getattr(prod, attr))
        else:
            methods_map[compsep_run["method"]](config, input_alms, compsep_run, nsim=nsim)

        if "mask" in compsep_run:
            del compsep_run['mask']

    if mask is not None:
        if remove_mask_type:
            delattr(config, 'mask_type')

    if config.return_compsep_products:
        for attr in vars(outputs).keys():
            setattr(outputs, attr, np.array(getattr(outputs, attr)))
        return outputs

def _estimate_residuals(config: Configs, nsim = None, mask = None):  # shape (N_sims, lmax - 1)
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

    if mask is not None:
        if not hasattr(config, "mask_type"):
            remove_mask_type = True
            config.mask_type = "mask_for_compsep"
        else:
            remove_mask_type = False
            
        if config.mask_type not in ["mask_for_compsep", "observed_patch"]:
            raise ValueError("Invalid value for 'mask_type'. It can be either 'mask_for_compsep' or 'observed_patch'.")
        else:
            if config.verbose:
                print(f"Provided mask is used as" + "observed patch" if config.mask_type == "observed_patch" else "mask for component separation.")

    if config.return_compsep_products:
        outputs = SimpleNamespace()
        for attr in vars(data):
            setattr(outputs, attr, [])
    
    if config.verbose:
        print(f"Running foregrounds residuals estimation for simulation {nsim}." if nsim is not None else f"Running component separation.")

    for compsep_run in config.compsep_fres_runs:
        data = get_gnilc_maps(config, compsep_run["gnilc_path"], nsim=nsim)

        if mask is not None:
            if (config.mask_type == "observed_patch"):
                mask_in = _preprocess_mask(mask, config.nside_in)
                input_alms = _alms_from_data(config, data, mask_in=mask_in, data_type="maps", bring_to_common_resolution=False, pixel_window_in=config.pixel_window_out)
            else:
                input_alms = _alms_from_data(config, data, data_type="maps", bring_to_common_resolution=False, pixel_window_in=config.pixel_window_out)
        else:
            input_alms = _alms_from_data(config, data, data_type="maps", bring_to_common_resolution=False, pixel_window_in=config.pixel_window_out)

        if mask is not None:
            compsep_run["mask"] = _preprocess_mask(mask, config.nside)

        if config.return_compsep_products:
            prod = get_residuals_template(config, input_alms, compsep_run, nsim=nsim)
            for attr in vars(prod).keys():
                getattr(outputs, attr).append(getattr(prod, attr))
        else:
            get_residuals_template(config, input_alms, compsep_run, nsim=nsim)

        if "mask" in compsep_run:
            del compsep_run['mask']

    if mask is not None:
        if remove_mask_type:
            delattr(config, 'mask_type')

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
    if compsep_run["domain"] == "pixel" or compsep_run["domain"] == "needlet":
        if "reduce_ilc_bias" not in compsep_run:
            compsep_run["reduce_ilc_bias"] = False
    
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
        if "clusters" not in compsep_run:
            raise ValueError("clusters must be provided for methods 'mcilc' and 'mc_ilc'.")
        if not isinstance(compsep_run["clusters"], str):
            raise ValueError("'clusters' must be a string - either 'ideal' or a path.")
        if compsep_run["clusters"] != "ideal":
            if not os.path.exists(compsep_run["clusters"])

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
    if config.data_type not in ["maps", "alms"]:
        raise ValueError("Invalid value for data_type. It must be 'maps' or 'alms'.")
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
                if config.field_out not in ["TQU", "TEB"]:
                    raise ValueError("Invalid value for field_out. It must be 'TQU' or 'TEB' for the provided data.")
        else:
            config.field_out = config.field_in
    return config

def _check_data_and_config(config: Configs, data):
    if not hasattr(data, "total"):
        raise ValueError("The provided data do not have the 'total' attribute.")
    
    allowed_attributes = ["total", "noise", "cmb", "fgds", "dust", "synch", "ame", "co"]
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


