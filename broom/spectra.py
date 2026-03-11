import healpy as hp
import re
import numpy as np
import os 
import fnmatch
import sys
from .routines import obj_out_to_array, _slice_outputs, _format_nsim, _log
from .configurations import Configs
from types import SimpleNamespace
from typing import Optional, Union, Dict, Any
from .saving import save_spectra, _save_mask, _get_full_path_out
try:
    import pymaster as nmt
except ImportError:
    print("Warning: NaMaster python package not found. Spectra computation, if requested with 'namaster', will not be available.")
    nmt = None

def _compute_spectra(config: Configs) -> Optional[SimpleNamespace]:

    """
    Compute the spectra from outputs of component separation.
    
    Parameters
    ----------
        config : Configs
            Configuration object containing parameters for spectra computation. It should include:
            - nsim_start: Scomponenttarting simulation number.
            - nsims: Number of simulations to compute spectra for.
            - compute_spectra: List of dictionaries containing parameters for spectra computation, including:
                - path_method: Path to the component separation outputs.
                - field_out: Fields of the ouputs to be loaded for spectra computation.
                - components_for_cls: List of components to compute spectra for.
                - mask_type: Type of mask to apply (e.g., 'GAL*+fgres', 'GAL*0', 'GAL97', 'GAL99', 'fgres', 'config+fgres', 'config').
                - apodize_mask: Type of apodization to apply to the mask (e.g., 'gaussian', 'C1'). If None, no apodization is applied.
                - smooth_mask: Apodization scale for the mask in degrees. Default is 5 degrees.
                - nmt_purify_B: Whether to purify B modes when using NaMaster. Default is True.
                - nmt_purify_E: Whether to purify E modes when using NaMaster. Default is False.
                - smooth_tracer: Smoothing scale for the tracer in degrees, used when mask_type contains 'fgres'. Default is 3 degrees.
                - fsky: Fraction of the sky to consider when computing the spectra, used when mask_type contains 'fgres'. Default is 1.0.
            - spectra_comp: Method to compute the spectra, either 'anafast' or 'namaster'.
            - return_Dell: Whether to return the spectra as D_ell or C_ell. Default is False (returns C_ell).
            - field_cls_out: Fields of the spectra to compute. 
            - save_spectra: Whether to save the computed spectra to files. Default is True.
            - path_outputs: Path where computed spectra will be saved.
            - return_spectra: Whether to return the computed spectra as a SimpleNamespace object. Default is True.
    
    Returns
    -------
        cls_ : SimpleNamespace
            Object containing computed spectra with attributes for each component.
            Each attribute is a numpy array with dimensions (nsim, ncases, nfields, nbins), 
            where ncases is the number of different component separation outputs provided in compute_spectra dictionary.
    """
    
    if not isinstance(config, Configs):
        raise TypeError("config must be an instance of Configs")

    cls_ = SimpleNamespace() if config.return_spectra else None

    for nsim in range(config.nsim_start, config.nsim_start + config.nsims):
        if config.return_spectra:
            cls_sim = _compute_spectra_(config, nsim=nsim)
            for attr, value in vars(cls_sim).items():
                if not hasattr(cls_, attr):
                    setattr(cls_, attr, [])
                getattr(cls_, attr).append(value)
        else:
            _compute_spectra_(config, nsim=nsim)

    if config.return_spectra:
        for attr in vars(cls_):
            setattr(cls_, attr, np.array(getattr(cls_, attr)))
        return cls_


def _compute_spectra_(
    config: Configs, 
    outputs: Optional[SimpleNamespace] = None,
    outputs_type: Optional[str] = None,
    nsim: Optional[Union[int, str]] = None
) -> Optional[SimpleNamespace]:
    """
    Compute the spectra from outputs of component separation for single simulation 'nsim'.

    Parameters
    ----------
        config : Configs
            Configuration object containing parameters for spectra computation. See `_compute_spectra` for details.
        outputs : SimpleNamespace, optional
            Object containing the outputs of component separation for a single simulation. If None, it will be loaded from disk based on 
            the paths specified in the compute_spectra dictionaries in config.
        outputs_type : str, optional
            Type of outputs provided in the outputs argument, either 'compsep', 'residuals_estimate' or 'compsep_propagate'.
            If None, it will be inferred from the lenght of the outputs object, if possible.
        nsim : int or str, optional
            Simulation number to compute spectra for. If an integer, it will be zero-padded to 5 digits.
            Default is None, which means it will look for outputs with no label regarding simulation number.

    Returns
    -------
        cls_out_ : SimpleNamespace
            Object containing computed spectra with attributes for each component.
            Each attribute is a numpy array with dimensions (ncases, nfields, nbins), 
            where ncases is the number of different component separation outputs provided in compute_spectra dictionary.
    """

    nsim = _format_nsim(nsim)

    if config.spectra_comp not in ['anafast','namaster']:
        raise ValueError('spectra_comp must be either "anafast" or "namaster"')

    cls_out_ = SimpleNamespace() if config.return_spectra else None

    load_outputs = outputs is None

    if not load_outputs:
        if outputs_type is not None:
            if outputs_type not in ['compsep', 'residuals_estimate', 'compsep_propagate']:
                raise ValueError('outputs_type must be either "compsep", "residuals_estimate" or "compsep_propagate"')
        else:
            outputs_type = 'compsep' if hasattr(config, 'compsep') and config.compsep is not None and len(config.compsep) > 0 else None
            if outputs_type is None:
                outputs_type = 'residuals_estimate' if hasattr(config, 'compsep_residuals') and config.compsep_residuals is not None and len(config.compsep_residuals) > 0 else None
                if outputs_type is None:
                    outputs_type = 'compsep_propagate' if hasattr(config, 'compsep_propagate') and config.compsep_propagate is not None and len(config.compsep_propagate) > 0 else None
                    if outputs_type is None:
                        raise ValueError('Could not infer outputs_type from config. Please provide outputs_type argument and make sure config contains either compsep, compsep_residuals or compsep_propagate attribute with non-empty list of dictionaries.')

        if outputs_type == 'compsep':
            if not hasattr(config, 'compsep') or config.compsep is None or len(config.compsep) == 0:
                raise ValueError('If outputs_type is "compsep", config must contain a non-empty compsep attribute.')
            idx_runs = []
            for i in range(len(config.compsep)):
                if config.compsep[i]["method"] not in ["fgd_diagnostic", "fgd_P_diagnostic"]:
                    idx_runs.append(i)
        elif outputs_type == 'residuals_estimate':
            if not hasattr(config, 'compsep_residuals') or config.compsep_residuals is None or len(config.compsep_residuals) == 0:
                raise ValueError('If outputs_type is "residuals_estimate", config must contain a non-empty compsep_residuals attribute.')
            idx_runs = range(len(config.compsep_residuals))
        elif outputs_type == 'compsep_propagate':
            if not hasattr(config, 'compsep_propagate') or config.compsep_propagate is None or len(config.compsep_propagate) == 0:
                raise ValueError('If outputs_type is "compsep_propagate", config must contain a non-empty compsep_propagate attribute.')
            idx_runs = range(len(config.compsep_propagate))
    
        n_runs = len(idx_runs)
        if (n_runs > 1) and (len(config.compute_spectra) > 1):
            if n_runs != len(config.compute_spectra):
                raise ValueError(f"If number of runs in outputs ({n_runs}) and of compute_spectra dictionaries in config ({len(config.compute_spectra)}) are larger than 1, they must match.")
        idx_cs = 0
        
    if load_outputs or (n_runs == 1) or (len(config.compute_spectra) > 1):
        for compute_cls in config.compute_spectra:
            compute_cls = _standardize_compute_cls(config, compute_cls, load_outputs=load_outputs)
            if not load_outputs:
                compute_cls["outputs_type"] = outputs_type

            if config.return_spectra:
                if load_outputs:
                    cls_out = _cls_from_config(config, compute_cls, nsim=nsim, outputs=outputs)
                else:
                    cls_out = _cls_from_config(config, compute_cls, nsim=nsim, outputs=outputs, idx_out=idx_runs[idx_cs], idx_cs=idx_cs)
                    if n_runs > 1:
                        idx_cs += 1
                for attr, value in vars(cls_out).items():
                    if not hasattr(cls_out_, attr):
                        setattr(cls_out_, attr, [])
                    getattr(cls_out_, attr).append(value)
            else:
                if load_outputs:
                    _cls_from_config(config, compute_cls, nsim=nsim, outputs=outputs)
                else:
                    _cls_from_config(config, compute_cls, nsim=nsim, outputs=outputs, idx_out=idx_runs[idx_cs], idx_cs=idx_cs)
                    if n_runs > 1:
                        idx_cs += 1
            compute_cls.pop("outputs_type", None)
    else:
        delete_components = "components_for_cls" not in config.compute_spectra[0] or config.compute_spectra[0]["components_for_cls"] is None
        for _ in range(n_runs):
            compute_cls = _standardize_compute_cls(config, config.compute_spectra[0], load_outputs=load_outputs)
            compute_cls["outputs_type"] = outputs_type 

            if config.return_spectra:
                cls_out = _cls_from_config(config, compute_cls, nsim=nsim, outputs=outputs, idx_out=idx_runs[idx_cs], idx_cs=idx_cs)
                for attr, value in vars(cls_out).items():
                    if not hasattr(cls_out_, attr):
                        setattr(cls_out_, attr, [])
                    getattr(cls_out_, attr).append(value)
            else:
                _cls_from_config(config, compute_cls, nsim=nsim, outputs=outputs, idx_out=idx_runs[idx_cs], idx_cs=idx_cs)
            idx_cs += 1

            compute_cls.pop("outputs_type", None)
            if delete_components:
                compute_cls.pop("components_for_cls", None)

    if config.return_spectra:
        for attr in vars(cls_out_):
            setattr(cls_out_, attr, np.array(getattr(cls_out_, attr)))
        return cls_out_

def _standardize_compute_cls(config: Configs, compute_cls: Dict[str, Any], load_outputs: bool = True) -> Dict[str, Any]:
    """
    Standardize the compute_cls dictionary to ensure it contains all necessary fields and has the correct structure.

    Parameters
    ----------
        config : Configs
            Configuration object containing parameters for spectra computation. See `_compute_spectra` for details.
        compute_cls : dict
            Dictionary containing parameters for spectra computation. See `_compute_spectra` for details.
        load_outputs : bool
            Whether the outputs of component separation need to be loaded from disk. 

    Returns
    -------
        compute_cls : dict
            Standardized dictionary containing parameters for spectra computation.
    """
    if load_outputs:
        if 'path_method' not in compute_cls:
            raise ValueError('compute_cls must contain a "path_method" field, since outputs need to be loaded from disk.')
        if "components_for_cls" not in compute_cls:
            raise ValueError('compute_cls must contain a "components_for_cls" field')
        compute_cls.setdefault('field_out', config.field_out)
    else:
        compute_cls["field_out"] = config.field_out
        
    compute_cls.setdefault("mask_type", None)
    compute_cls.setdefault("apodize_mask", None)

    if compute_cls["apodize_mask"] is not None:
        compute_cls.setdefault("smooth_mask", 5.)

    if compute_cls["mask_type"] is not None and "fgres" in compute_cls["mask_type"]:
        compute_cls.setdefault("smooth_tracer", 3.0)
        if "fsky" not in compute_cls:
            compute_cls["fsky"] = 1.
            print("'fsky' not defined in compute_spectra; set to 1.0 (no fg residual thresholding).")
    if config.spectra_comp == 'namaster':
        compute_cls.setdefault("nmt_purify_B", True)
        compute_cls.setdefault("nmt_purify_E", False)

    return compute_cls
    
def _cls_from_config(
    config: Configs,
    compute_cls: Dict[str, Any],
    nsim: Optional[str] = None,
    outputs: Optional[SimpleNamespace] = None,
    idx_out: Optional[int] = None,
    idx_cs: Optional[int] = None
) -> Optional[SimpleNamespace]:
    """
    Compute the spectra from outputs of component separation based on the configuration and compute_cls dictionary.
    
    Parameters
    ----------
        config : Configs
            Configuration object containing parameters for spectra computation. See `_compute_spectra` for details.
        compute_cls : dict
            Dictionary containing parameters for spectra computation. See `_compute_spectra` for details.
        nsim : str, optional
            Simulation number. Used to load the correct outputs. If None, it will look for outputs with no label regarding simulation number.
        outputs : SimpleNamespace, optional
            Object containing the outputs of component separation for a single simulation. If None, it will be loaded from disk based on 
            the paths specified in the compute_cls dictionary in config.
        idx_out : int, optional
            Index of the output to use from the configuration object, if outputs is provided. This is relevant when outputs contains multiple runs of component separation.
        idx_cs : int, optional
            Index of the component separation run to use from the outputs object, if outputs is provided and contains multiple runs. This is relevant when outputs contains multiple runs of component separation.
    
    Returns
    -------
        cls_out : SimpleNamespace
            Object containing computed spectra with attributes for each component.
            Each attribute is a numpy array with dimensions (nfields, nbins), 
    """
    from .compsep import _load_outputs_

    compute_cls["outputs"] = SimpleNamespace()  

    _check_fields_for_cls(compute_cls["field_out"],config.field_cls_out)
    compute_cls["field_cls_in"] = _get_fields_in_for_cls(compute_cls["field_out"],config.field_cls_out)
    
    if outputs is None:
        if "path_outputs" in compute_cls:
            compute_cls["path"] = os.path.join(compute_cls["path_outputs"], compute_cls["path_method"])
        else:
            compute_cls["path"] = os.path.join(config.path_outputs, compute_cls["path_method"])

        for component in compute_cls["components_for_cls"]:
            if isinstance(component, str):
                if "gilc_" in compute_cls["path"] or "gpilc_" in compute_cls["path"] or "gprilc_" in compute_cls["path"]:
                    component_name = '_'.join(component.split('_')[:-1])
                    if nsim is not None:
                        filename = os.path.join(
                            compute_cls["path"],
                            f"{component_name}/{nsim}/{compute_cls['field_out']}_{component_name}_{component.split('_')[-1]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
                        )
                    else:
                        filename = os.path.join(
                            compute_cls["path"],
                            f"{component_name}/{compute_cls['field_out']}_{component_name}_{component.split('_')[-1]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
                        )
                    component_name = component
                else:
                    component_name = component.split('/')[0] if '/' in component else component
                    filename = os.path.join(
                        compute_cls["path"],
                        f"{component}/{compute_cls['field_out']}_{component_name}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
                    )

                setattr(
                    compute_cls["outputs"],
                    component_name,
                    _load_outputs_(filename, compute_cls["field_out"], nsim=nsim)
                )

            elif isinstance(component, list):
                if len(component)==2:
                    filenames = []
                    for idx, component_ in enumerate(component):
                        if "gilc_" in compute_cls["path"] or "gpilc_" in compute_cls["path"] or "gprilc_" in compute_cls["path"]:
                            component_name_ = '_'.join(component_.split('_')[:-1])
                            if nsim is not None:
                                filenames.append(os.path.join(
                                    compute_cls["path"],
                                    f"{component_name_}/{nsim}/{compute_cls['field_out']}_{component_name_}_{component_.split('_')[-1]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
                                ))
                            else:
                                filenames.append(os.path.join(
                                    compute_cls["path"],
                                    f"{component_name_}/{compute_cls['field_out']}_{component_name_}_{component_.split('_')[-1]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
                                ))
                            if idx == 0:
                                component_name = component_
                            elif idx == 1:
                                component_name += f"_x_{component_}"
                        else:
                            component_name_ = component_.split('/')[0] if '/' in component_ else component_
                            filenames.append(os.path.join(
                                compute_cls["path"],
                                f"{component_}/{compute_cls['field_out']}_{component_name_}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
                            ))
                            if idx == 0:
                                component_name = component_name_
                            elif idx == 1:
                                component_name += f"_x_{component_name_}"

                    setattr(
                        compute_cls["outputs"],
                        component_name,
                        [])
                    
                    for filename in filenames:
                        getattr(
                            compute_cls["outputs"],
                            component_name
                        ).append(_load_outputs_(filename, compute_cls["field_out"], nsim=nsim))

                    setattr(compute_cls["outputs"], component_name, np.array(getattr(
                        compute_cls["outputs"],
                        component_name
                    )))

                    if getattr(
                        compute_cls["outputs"],
                        component_name
                    ).ndim == 3:
                        setattr(
                            compute_cls["outputs"],
                            component_name,
                            np.transpose(getattr(
                                compute_cls["outputs"],
                                component_name
                            ), (1,0,2)
                        ))
                else:
                    raise ValueError("Components as list must contain exactly two elements for cross-spectra.")
            else:
                raise TypeError("Components for cls must be either str or list of two str.")
    
    else:
        if "components_for_cls" not in compute_cls:
            all_components = True
            compute_cls["components_for_cls"] = []
        else:
            all_components = False

        if compute_cls["outputs_type"] == 'compsep':
            compute_cls["path"] = _get_full_path_out(config, config.compsep[idx_out])

        elif compute_cls["outputs_type"] == 'residuals_estimate':
            compute_cls["path"] = os.path.join(config.path_outputs, config.compsep_residuals[idx_out]["compsep_path"])

            if "gilc_" in config.compsep_residuals[idx_out]["gilc_path"]:
                gnilc_run = (re.search(r'(gilc_[^/]+)', config.compsep_residuals[idx_out]["gilc_path"])).group(1)
            elif "gpilc_" in config.compsep_residuals[idx_out]["gilc_path"]:
                gnilc_run = (re.search(r'(gpilc_[^/]+)', config.compsep_residuals[idx_out]["gilc_path"])).group(1)        
            elif "gprilc_" in config.compsep_residuals[idx_out]["gilc_path"]:
                gnilc_run = (re.search(r'(gprilc_[^/]+)', config.compsep_residuals[idx_out]["gilc_path"])).group(1)        
            if "needlet" in gnilc_run:
                folder_after = (config.compsep_residuals[idx_out]["gilc_path"]).split(gnilc_run + "/")[1].split("/")[0]
                gnilc_run += f"_{folder_after}"

        elif compute_cls["outputs_type"] == 'compsep_propagate':
            compute_cls["path"] = os.path.join(config.path_outputs, config.compsep_propagate[idx_out]["compsep_path"])


        if all_components:
            for component, array in vars(outputs).items():
                if compute_cls["outputs_type"] == 'compsep':
                    if "total" in component:
                        component_name = f"output_{component}"
                    elif "cmb" in component:
                        if config.compsep[idx_out]["component_out"] == "cmb":
                            component_name = f"output_{component}"
                        else:
                            component_name = f"{component}_residuals"
                    elif "tsz" in component:
                        if config.compsep[idx_out]["component_out"] == "tsz":
                            component_name = f"output_{component}"
                        else:
                            component_name = f"{component}_residuals"
                    else:
                        component_name = f"{component}_residuals"
                elif compute_cls["outputs_type"] == 'compsep_propagate':
                    component_name = f"propagated_{component}"
                elif compute_cls["outputs_type"] == 'residuals_estimate':
                    if "total" in component:
                        component_name = f"fgres_templates"
                        if "split" in component:
                            component_name += f"_{component.split('total_')[1]}"
                    elif component == "fgds":
                        component_name = f"fgres_templates_fgds"
                    else:
                        component_name = f"fgres_templates_{component}"
                    component_name += f"/{gnilc_run}"

                array = np.array(array[idx_cs])

                if compute_cls["outputs_type"] == 'compsep':
                    if config.compsep[idx_out]["method"] in ["gilc", "gpilc", "gprilc"]:
                        for idx_f in range(array.shape[0]):
                            component_name_ = f"{component_name}_{config.instrument.channels_tags[config.compsep[idx_out]['channels_out'][idx_f]]}"
                            compute_cls["components_for_cls"].append(component_name_)
                            setattr(compute_cls["outputs"], component_name_, array[idx_f])
                    else:
                        compute_cls["components_for_cls"].append(component_name)
                        setattr(compute_cls["outputs"], component_name, array)
                elif compute_cls["outputs_type"] == 'compsep_propagate':
                    if "gilc_" in config.compsep_propagate[idx_out]["compsep_path"] or "gpilc_" in config.compsep_propagate[idx_out]["compsep_path"] or "gprilc_" in config.compsep_propagate[idx_out]["compsep_path"]:
                        for idx_f in range(array.shape[0]):
                            component_name_ = f"{component_name}_{config.instrument.channels_tags[config.compsep_propagate[idx_out]['channels_out'][idx_f]]}"
                            compute_cls["components_for_cls"].append(component_name_)
                            setattr(compute_cls["outputs"], component_name_, array[idx_f])
                    else:
                        compute_cls["components_for_cls"].append(component_name)
                        setattr(compute_cls["outputs"], component_name, array)
                elif compute_cls["outputs_type"] == 'residuals_estimate':
                    compute_cls["components_for_cls"].append(component_name)
                    setattr(compute_cls["outputs"], component_name.split('/')[0] if '/' in component_name else component_name, array)
        else:
            for component in compute_cls["components_for_cls"]:
                if isinstance(component, str):
                    if compute_cls["outputs_type"] == 'compsep':
                        if "total" in component:
                            component_name = "total"
                            if "split1" in component:
                                component_name += "_split1"
                            elif "split2" in component:
                                component_name += "_split2"
                        elif "cmb" in component:
                            component_name = f"cmb"
                        elif "tsz" in component:
                            component_name = f"tsz"
                        else:
                            component_name = component.split('_residuals')[0]
                    elif compute_cls["outputs_type"] == 'compsep_propagate':
                        component_name = component.split('propagated_')[1]
                        if "gilc_" in config.compsep_propagate[idx_out]["compsep_path"] or "gpilc_" in config.compsep_propagate[idx_out]["compsep_path"] or "gprilc_" in config.compsep_propagate[idx_out]["compsep_path"]:
                            if component.split('_')[-1] in config.instrument.channels_tags:
                                component_name = component_name.split(f"_{component_name.split('_')[-1]}")[0]
                    elif compute_cls["outputs_type"] == 'residuals_estimate':
                        idx_component = compute_cls["components_for_cls"].index(component)
                        if "/" in component:
                            component = component.split('/')[0]
                        if component == "fgres_templates":
                            component_name = "total"
                        elif component == "fgres_templates_fgds":
                            component_name = "fgds"
                        elif "fgres_templates_split" in component:
                            component_name = f"total_{component.split('fgres_templates_')[1]}"
                        else:
                            component_name = component.split('fgres_templates_')[1]

                    array = np.array(getattr(outputs, component_name)[idx_cs])

                    if compute_cls["outputs_type"] == 'compsep':
                        if config.compsep[idx_out]["method"] in ["gilc", "gpilc", "gprilc"]:
                            if component.split('_')[-1] in config.instrument.channels_tags:
                                idx_f = config.instrument.channels_tags.index(component.split('_')[-1])
                                if idx_f in config.compsep[idx_out]['channels_out']:
                                    setattr(compute_cls["outputs"], component, array[config.compsep[idx_out]['channels_out'].index(idx_f)])
                                else:
                                    raise ValueError(f"Channel {component.split('_')[-1]} not found in channels_out of config.compsep[{idx_out}].")
                            else:
                                raise ValueError(f"Component {component} in compute_cls['components_for_cls'] is not in the correct format for method {config.compsep[idx_out]['method']}.")
#                                for idx_f in range(array.shape[0]):
#                                    component_name = f"{component}_{config.instrument.channels_tags[config.compsep[idx_out]['channels_out'][idx_f]]}"
#                                    setattr(compute_cls["outputs"], component_name, array[idx_f])
                        else:
                            setattr(compute_cls["outputs"], component, array)
                    elif compute_cls["outputs_type"] == 'compsep_propagate':
                        if "gilc_" in config.compsep_propagate[idx_out]["compsep_path"] or "gpilc_" in config.compsep_propagate[idx_out]["compsep_path"] or "gprilc_" in config.compsep_propagate[idx_out]["compsep_path"]:
                            if component.split('_')[-1] in config.instrument.channels_tags:
                                idx_f = config.instrument.channels_tags.index(component.split('_')[-1])
                                if idx_f in config.compsep_propagate[idx_out]['channels_out']:
                                    setattr(compute_cls["outputs"], component, array[config.compsep_propagate[idx_out]['channels_out'].index(idx_f)])
                                else:
                                    raise ValueError(f"Channel {component.split('_')[-1]} not found in channels_out of config.compsep_propagate[{idx_out}].")
                            else:
                                raise ValueError(f"Component {component} in compute_cls['components_for_cls'] is not in the correct format for method {config.compsep_propagate[idx_out]['method']}.")
#                            else:
#                                for idx_f in range(array.shape[0]):
#                                    component_name = f"{component}_{config.instrument.channels_tags[config.compsep_propagate[idx_out]['channels_out'][idx_f]]}"
#                                    setattr(compute_cls["outputs"], component_name, array[idx_f])
                        else:
                            setattr(compute_cls["outputs"], component, array)
                    elif compute_cls["outputs_type"] == 'residuals_estimate':
                        compute_cls["components_for_cls"][idx_component] = f"{component}/{gnilc_run}"
                        setattr(compute_cls["outputs"], f"{component}", array)

                elif isinstance(component, list):
                    if len(component)==2:
                        if '/' in component[0]:
                            component[0] = component[0].split('/')[0]
                        if '/' in component[1]:
                            component[1] = component[1].split('/')[0]

                        component_name = f"{component[0]}_x_{component[1]}"
                        setattr(
                            compute_cls["outputs"],
                            component_name,
                            []
                        )

                        for component_ in component:
                            if compute_cls["outputs_type"] == 'compsep':
                                if "total" in component_:
                                    component_name_ = "total"
                                    if "split1" in component_:
                                        component_name_ += "_split1"
                                    elif "split2" in component_:
                                        component_name_ += "_split2"
                                elif "cmb" in component_:
                                    component_name_ = f"cmb"
                                elif "tsz" in component_:
                                    component_name_ = f"tsz"
                                else:
                                    component_name_ = component_.split('_residuals')[0]
                            elif compute_cls["outputs_type"] == 'compsep_propagate':
                                component_name_ = component_.split('propagated_')[1]
                                if "gilc_" in config.compsep_propagate[idx_out]["compsep_path"] or "gpilc_" in config.compsep_propagate[idx_out]["compsep_path"] or "gprilc_" in config.compsep_propagate[idx_out]["compsep_path"]:
                                    if component_.split('_')[-1] in config.instrument.channels_tags:
                                        component_name_ = component_name_.split(f"_{component_name_.split('_')[-1]}")[0]
                            elif compute_cls["outputs_type"] == 'residuals_estimate':
                                if component_ == "fgres_templates":
                                    component_name_ = "total"
                                elif component_ == "fgres_templates_fgds":
                                    component_name_ = "fgds"
                                elif "fgres_templates_split" in component_:
                                    component_name_ = f"total_{component_.split('fgres_templates_')[1]}"
                                else:
                                    component_name_ = component_.split('fgres_templates_')[1]
                                
                            array = np.array(getattr(outputs, component_name_)[idx_cs])

                            if compute_cls["outputs_type"] == 'compsep':
                                if config.compsep[idx_out]["method"] in ["gilc", "gpilc", "gprilc"]:
                                    if component_.split('_')[-1] in config.instrument.channels_tags:
                                        idx_f = config.instrument.channels_tags.index(component_.split('_')[-1])
                                        if idx_f in config.compsep[idx_out]['channels_out']:
                                            getattr(compute_cls["outputs"], component_name).append(array[config.compsep[idx_out]['channels_out'].index(idx_f)])
                                        else:
                                            raise ValueError(f"Channel {component_.split('_')[-1]} not found in channels_out of config.compsep[{idx_out}].")
                                    else:
                                        raise ValueError(f"Component {component_} in compute_cls['components_for_cls'] is not in the correct format for method {config.compsep[idx_out]['method']}.")
                                else:
                                    getattr(compute_cls["outputs"], component_name).append(array)
                            elif compute_cls["outputs_type"] == 'compsep_propagate':
                                if "gilc_" in config.compsep_propagate[idx_out]["compsep_path"] or "gpilc_" in config.compsep_propagate[idx_out]["compsep_path"] or "gprilc_" in config.compsep_propagate[idx_out]["compsep_path"]:
                                    if component_.split('_')[-1] in config.instrument.channels_tags:
                                        idx_f = config.instrument.channels_tags.index(component_.split('_')[-1])
                                        if idx_f in config.compsep_propagate[idx_out]['channels_out']:
                                            getattr(compute_cls["outputs"], component_name).append(array[config.compsep_propagate[idx_out]['channels_out'].index(idx_f)])
                                        else:
                                            raise ValueError(f"Channel {component_.split('_')[-1]} not found in channels_out of config.compsep_propagate[{idx_out}].")
                                    else:
                                        raise ValueError(f"Component {component_} in compute_cls['components_for_cls'] is not in the correct format for method {config.compsep_propagate[idx_out]['method']}.")
                                else:
                                    getattr(compute_cls["outputs"], component_name).append(array)
                            elif compute_cls["outputs_type"] == 'residuals_estimate':
                                getattr(compute_cls["outputs"], component_name).append(array)

                        setattr(compute_cls["outputs"], component_name, np.array(getattr(compute_cls["outputs"], component_name)))
                        if getattr(
                            compute_cls["outputs"],
                            component_name
                        ).ndim == 3:
                            setattr(
                                compute_cls["outputs"],
                                component_name,
                                np.transpose(getattr(
                                    compute_cls["outputs"],
                                    component_name
                                ), (1,0,2)
                            ))
                    else:
                        raise ValueError("Elements in components_for_cls as list must either be strings or lists of two strings for cross-spectra.")

    #if (config.field_out in ["TQU", "TEB"]) and (config.field_cls_out not in ["TE", "TB", "TEB"]):
    if len(compute_cls['field_out']) > 1:
        compute_cls["outputs"] = _slice_outputs(
            compute_cls["outputs"],
            compute_cls["field_out"],
            compute_cls["field_cls_in"]
        )

    cls_out = _cls_from_maps(config, compute_cls, nsim=nsim)

    if config.save_mask:
        if compute_cls["mask_type"] is not None:
            #if 'fgres' in compute_cls["mask_type"] or 'fgtemp' in compute_cls["mask_type"]:
            _save_mask(compute_cls["mask"], config, compute_cls, nsim=nsim)
            #else:
            #    _log("Mask type does not include 'fgres' or 'fgtemp', not saving mask.", verbose=config.verbose)
        else:
            _log("Mask type not defined, not saving mask.", verbose=config.verbose)

    if config.save_spectra:
        save_spectra(config, cls_out, compute_cls, nsim=nsim)

    # Clean up
    compute_cls.pop("outputs", None)
    compute_cls.pop("path", None)
    compute_cls.pop("mask", None)

    if config.return_spectra:
        return cls_out

def _cls_from_maps(
    config: Configs,
    compute_cls: Dict[str, Any],
    nsim: Optional[str] = None
) -> SimpleNamespace:
    """
    Compute and return the spectra from maps stored in compute_cls dictionary after generating the mask.

    Parameters
    ----------
        config : Configs
            Configuration object containing global settings for spectra computation. See `_compute_spectra` for details.
        compute_cls : dict
            Dictionary containing parameters for spectra computation. See `_compute_spectra` for details.
            It must include the attribute `outputs`, which is a SimpleNamespace containing the maps for each component.
        nsim : str, optional
            Simulation number associated to the provided output maps.
    
    Returns
    -------
        cls_out : SimpleNamespace
            Object containing computed spectra with attributes for each component.
            Each attribute is a numpy array with dimensions (nfields, nbins),
    """
    from .masking import _get_mask, _smooth_masks

    compute_cls["mask"] = _get_mask(config, compute_cls, nsim=nsim)

    if not (compute_cls["mask_type"] is None and config.mask_observations is None and config.mask_covariance is None):
        if compute_cls["apodize_mask"] is not None:
            compute_cls["mask"] = _smooth_masks(
                compute_cls["mask"],
                compute_cls["apodize_mask"],
                compute_cls["smooth_mask"]
            )

    cls_out = _get_cls(config, compute_cls, nsim=nsim)

    return cls_out

def _get_cls(config: Configs, compute_cls, nsim=None):
    """
    Compute the spectra from the maps stored in compute_cls dictionary and using the provided mask.

    Parameters
    ----------
        config : Configs
            Configuration object containing parameters for spectra computation. See `_compute_spectra` for details.
        compute_cls : dict
            Dictionary containing parameters for spectra computation. See `_compute_spectra` for details.
        nsim : str, optional
            Simulation number associated to the provided output maps. If None, it will look for outputs with no label regarding simulation number.
    
    Returns
    -------
        cls_out : SimpleNamespace
            Object containing computed spectra with attributes for each component.
            Each attribute is a numpy array with dimensions (nfields, nbins),
    """
    from .masking import get_masks_for_compsep

    if config.ell_min_bpws is not None:
        ell_ini = np.array(config.ell_min_bpws)
        ell_end = ell_ini[1:]
        ell_end = np.append(ell_end, config.lmax+1)

    if config.spectra_comp == 'namaster':
        if config.ell_min_bpws is not None:
            b_bin = nmt.NmtBin.from_edges(ell_ini, ell_end, is_Dell=config.return_Dell)  
        else:
            b_bin = nmt.NmtBin.from_lmax_linear(config.lmax, nlb=config.delta_ell,is_Dell=config.return_Dell)
        wspaces = {}
    elif config.spectra_comp == 'anafast':
        if config.ell_min_bpws is not None:
            b_bin = BBin.from_edges(ell_ini, ell_end, is_Dell=config.return_Dell)  
        else:
            b_bin = BBin.from_lmax_linear(config.lmax, nlb=config.delta_ell,is_Dell=config.return_Dell)

    bls_beam = get_bls(config.nside, config.fwhm_out, config.lmax, config.field_cls_out, pixel_window_out=config.pixel_window_out)
    
    cls_out = SimpleNamespace()
    for attr in vars(compute_cls["outputs"]):
        setattr(cls_out, attr, [])

#    ndim = obj_out_to_array(compute_cls["outputs"]).ndim
#    out_shapes = obj_out_to_array(compute_cls["outputs"]).shape

    def compute_anafast_scalar(maps, mask, beam):
        cl = hp.anafast(maps * mask, lmax=config.lmax, pol=False)
        cl /= np.mean(mask**2) * beam**2
        if mask_in_maps is not None:
            cl /= np.mean(mask_in_maps**2)
        return b_bin.bin_cell(cl)

    def compute_namaster_scalar(maps, mask, beam, field1):
        f = nmt.NmtField(mask, [maps], beam=beam, lmax=config.lmax, lmax_mask=config.lmax)
        if mask_in_maps is not None:
            if f"00_{field1}{field1}" not in wspaces:
                f_w = nmt.NmtField(mask * mask_in_maps, [maps], beam=beam, lmax=config.lmax, lmax_mask=config.lmax)
                wspaces[f"00_{field1}{field1}"] = nmt.NmtWorkspace.from_fields(f_w, f_w, b_bin)
        else:
            if f"00_{field1}{field1}" not in wspaces:
                wspaces[f"00_{field1}{field1}"] = nmt.NmtWorkspace.from_fields(f, f, b_bin)
        return (nmt.compute_full_master(f, f, b_bin, workspace=wspaces[f"00_{field1}{field1}"]))[0]
        
    def compute_anafast_cross_scalars(map1, map2, mask, beam1, beam2):
        cl_cross = hp.anafast(map1 * mask, map2=map2 * mask,lmax=config.lmax,pol=False)
        cl_cross /= np.mean(mask**2) * (beam1 * beam2)
        if mask_in_maps is not None:
            cl_cross /= np.mean(mask_in_maps**2)
        return b_bin.bin_cell(cl_cross)

    def compute_namaster_cross_scalars(map1, map2, mask1, mask2, beam1, beam2, field1, field2):
        f1 = nmt.NmtField(mask1, [map1],beam=beam1,lmax=config.lmax,lmax_mask=config.lmax)
        f2 = nmt.NmtField(mask2, [map2],beam=beam2,lmax=config.lmax,lmax_mask=config.lmax)
        if mask_in_maps is not None:
            f1_w = nmt.NmtField(mask1 * mask_in_maps, [map1],beam=beam1,lmax=config.lmax,lmax_mask=config.lmax)
            f2_w = nmt.NmtField(mask2 * mask_in_maps, [map2],beam=beam2,lmax=config.lmax,lmax_mask=config.lmax)
            if f"00_{field1}{field2}" not in wspaces:
                wspaces[f"00_{field1}{field2}"] = nmt.NmtWorkspace.from_fields(f1_w, f2_w, b_bin)        
        else:
            if f"00_{field1}{field2}" not in wspaces:
                wspaces[f"00_{field1}{field2}"] = nmt.NmtWorkspace.from_fields(f1, f2, b_bin)
        return (nmt.compute_full_master(f1, f2, b_bin, workspace=wspaces[f"00_{field1}{field2}"]))[0]

    def compute_anafast_full_TQU(T_map, Q_map, U_map, T_mask, Q_mask, beam_T, beam_E, beam_B, get_cross=False):
        if T_map.ndim == 1 and Q_map.ndim ==1 and U_map.ndim ==1:
            cl = hp.anafast([T_map * T_mask, Q_map * Q_mask, U_map * Q_mask], lmax=config.lmax, pol=True)[:3]
        elif T_map.ndim == 2 and Q_map.ndim ==2 and U_map.ndim ==2:
            cl = hp.anafast([T_map[0] * T_mask, Q_map[0] * Q_mask, U_map[0] * Q_mask], 
                map2=[T_map[1] * T_mask, Q_map[1] * Q_mask, U_map[1] * Q_mask], lmax=config.lmax, pol=True)[:3]
        
        cl[0] = cl[0] / np.mean(T_mask**2) / (beam_T**2)
        cl[1] = cl[1] / np.mean(Q_mask**2) / (beam_E**2)
        cl[2] = cl[2] / np.mean(Q_mask**2) / (beam_B**2)
        if mask_in_maps is not None:
            cl /= np.mean(mask_in_maps**2)
        if get_cross:
            mask_ =  Q_mask if np.mean(np.ceil(T_mask)) > np.mean(np.ceil(Q_mask)) else T_mask
            if T_map.ndim == 1 and Q_map.ndim ==1 and U_map.ndim ==1:
                cl_cross = (hp.anafast([T_map * mask_, Q_map * mask_, U_map * mask_], lmax=config.lmax, pol=True) / np.mean(mask_**2))[3:]
            elif T_map.ndim == 2 and Q_map.ndim ==2 and U_map.ndim ==2:
                cl_cross = (hp.anafast([T_map[0] * mask_, Q_map[0] * mask_, U_map[0] * mask_], 
                    map2=[T_map[1] * mask_, Q_map[1] * mask_, U_map[1] * mask_], lmax=config.lmax, pol=True) / np.mean(mask_**2))[3:]
            cl_cross[0] = cl_cross[0] / (beam_T * beam_E)
            cl_cross[1] = cl_cross[1] / (beam_E * beam_B)
            cl_cross[2] = cl_cross[2] / (beam_T * beam_B)
            if mask_in_maps is not None:
                cl_cross /= np.mean(mask_in_maps**2)
            return np.concatenate([b_bin.bin_cell(cl), b_bin.bin_cell(cl_cross)], axis=0)
        return b_bin.bin_cell(cl)

    
    if 'purify' in compute_cls["path"]: # or 'maskonly' in compute_cls["path"]:
        _log('Output maps have been weighted by config mask. This will be taken into account.', verbose=config.verbose)
#        mask_in_maps = _preprocess_mask(hp.read_map(config.mask_path, field=0), config.nside)
        mask_in_maps, _ = get_masks_for_compsep(config.mask_observations, config.mask_covariance, config.nside)
        mask_in_maps /= np.max(mask_in_maps)
    else:
        mask_in_maps = None

    for idx, attr in enumerate(vars(compute_cls["outputs"])):
        output_data = getattr(compute_cls["outputs"], attr)

        if '_x_' in attr:
            if output_data.ndim == 2:
                if config.spectra_comp == 'anafast':
                    getattr(cls_out, attr).append(compute_anafast_cross_scalars(output_data[0], output_data[1], compute_cls["mask"], compute_cls["mask"], bls_beam[0], bls_beam[0]))
                else:
                    getattr(cls_out, attr).append(compute_namaster_cross_scalars(output_data[0], output_data[1], compute_cls["mask"], compute_cls["mask"], bls_beam[0], bls_beam[0], compute_cls["field_cls_in"], compute_cls["field_cls_in"]))
            else:
                if 'QU' not in compute_cls["field_cls_in"]:
                    for field in range(output_data.shape[0]):
                        if config.spectra_comp == 'anafast':
                            getattr(cls_out, attr).append(compute_anafast_cross_scalars(
                                output_data[field,0],  output_data[field,1], compute_cls["mask"][field], compute_cls["mask"][field], 
                                bls_beam[field], bls_beam[field]))
                        else:
                            getattr(cls_out, attr).append(compute_namaster_cross_scalars(
                            output_data[field,0], output_data[field,1], compute_cls["mask"][field], compute_cls["mask"][field],
                            bls_beam[field], bls_beam[field], compute_cls["field_cls_in"][field], compute_cls["field_cls_in"][field]
                            ))
                    
                    if "EETE" in config.field_cls_out or "BBTE" in config.field_cls_out:
                        if config.spectra_comp == 'anafast':
                            mask_ =  compute_cls["mask"][1] if np.mean(np.ceil(compute_cls["mask"][0])) > np.mean(np.ceil(compute_cls["mask"][1])) else compute_cls["mask"][0]
                            getattr(cls_out, attr).append(compute_anafast_cross_scalars(
                                output_data[0,0], output_data[1,1], mask_, bls_beam[0], bls_beam[1]))
                        elif config.spectra_comp == 'namaster':                        
                            getattr(cls_out, attr).append(compute_namaster_cross_scalars(
                                output_data[0,0], output_data[1,1], compute_cls["mask"][0], compute_cls["mask"][1],
                                bls_beam[0], bls_beam[1], compute_cls["field_cls_in"][0], compute_cls["field_cls_in"][1]
                            ))
                    
                    if "BBEB" in config.field_cls_out or "BTEEB" in config.field_cls_out:
                        field_E, field_B = (1, 2) if "T" in compute_cls["field_cls_in"] else (0, 1)
                        if config.spectra_comp == 'anafast':
                            mask_ =  compute_cls["mask"][field_B] if np.mean(np.ceil(compute_cls["mask"][field_E])) > np.mean(np.ceil(compute_cls["mask"][field_B])) else compute_cls["mask"][field_E]
                            getattr(cls_out, attr).append(
                                compute_anafast_cross_scalars(output_data[field_E,0], output_data[field_B,1], 
                                mask_, bls_beam[field_E], bls_beam[field_B]))
                        elif config.spectra_comp == 'namaster':
                            getattr(cls_out, attr).append(compute_namaster_cross_scalars(
                                output_data[field_E,0], output_data[field_B,1], compute_cls["mask"][field_E], compute_cls["mask"][field_B],
                                bls_beam[field_E], bls_beam[field_B], compute_cls["field_cls_in"][field_E], compute_cls["field_cls_in"][field_B]
                            ))

                    if "BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out:
                        field_B = 2 if "E" in compute_cls["field_cls_in"] else 1
                        if config.spectra_comp == 'anafast':
                            mask_ =  compute_cls["mask"][field_B] if np.mean(np.ceil(compute_cls["mask"][0])) > np.mean(np.ceil(compute_cls["mask"][field_B])) else compute_cls["mask"][0]
                            getattr(cls_out, attr).append(compute_anafast_cross_scalars(
                                output_data[0,0], output_data[field_B,1], mask_, bls_beam[0], bls_beam[field_B]))
                        elif config.spectra_comp == 'namaster':
                            getattr(cls_out, attr).append(compute_namaster_cross_scalars(
                                output_data[0,0], output_data[field_B,1], compute_cls["mask"][0], compute_cls["mask"][field_B],
                                bls_beam[0], bls_beam[field_B], compute_cls["field_cls_in"][0], compute_cls["field_cls_in"][field_B]
                            ))
                else:
                    if "T" in compute_cls["field_cls_in"]:
                        field_Q = 1
                        field_B = 1 if "EE" not in config.field_cls_out else 2 if "BB" in config.field_cls_out else None
                        field_E = 1 if "EE" in config.field_cls_out else None
                        beam_nmt = bls_beam[0]
                        T_map, Q_map, U_map = output_data[0], output_data[1], output_data[2]
                    else:
                        mask_ =  compute_cls["mask"][0]
                        field_Q = 0
                        field_B = 0 if "EE" not in config.field_cls_out else 1 if "BB" in config.field_cls_out else None
                        field_E = 0 if "EE" in config.field_cls_out else None
                        beam_nmt = hp.gauss_beam(np.radians(config.fwhm_out/60.), lmax=config.lmax, pol=False)
                        if config.pixel_window_out:
                            beam_nmt *= hp.pixwin(config.nside, lmax=config.lmax, pol=False)
                        T_map, Q_map, U_map = np.zeros_like(output_data[0]), output_data[0], output_data[1]

                    get_cross = any(x in config.field_cls_out for x in ["EETE", "BBTE", "BBEB", "BTEEB", "BBTB", "EBTB"])
                    if config.spectra_comp == 'anafast':
                        cls_s2 = compute_anafast_full_TQU(
                            T_map, Q_map, U_map,
                            compute_cls["mask"][0], compute_cls["mask"][field_Q],
                            bls_beam[0], bls_beam[field_E], bls_beam[field_B], get_cross=get_cross
                        )
                        if "TT" in config.field_cls_out:
                            getattr(cls_out, attr).append(cls_s2[0])
                        if "EE" in config.field_cls_out:
                            getattr(cls_out, attr).append(cls_s2[1])
                        if "BB" in config.field_cls_out:
                            getattr(cls_out, attr).append(cls_s2[2])
                        if get_cross:
                            if "EETE" in config.field_cls_out or "BBTE" in config.field_cls_out:
                                getattr(cls_out, attr).append(cls_s2[3])
                            if "BBEB" in config.field_cls_out or "BTEEB" in config.field_cls_out:
                                getattr(cls_out, attr).append(cls_s2[4])
                            if "BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out:
                                getattr(cls_out, attr).append(cls_s2[5])
                    elif config.spectra_comp == 'namaster':
                        if "TT" in config.field_cls_out:
                            f_0_0 = nmt.NmtField(compute_cls["mask"][0], [T_map[0]], beam=bls_beam[0],lmax=config.lmax,lmax_mask=config.lmax)
                            f_0_1 = nmt.NmtField(compute_cls["mask"][0], [T_map[1]], beam=bls_beam[0],lmax=config.lmax,lmax_mask=config.lmax)
                            if idx==0:
                                if mask_in_maps is not None:
                                    f_0_0_w = nmt.NmtField(compute_cls["mask"][0] * mask_in_maps, [T_map[0]], beam=bls_beam[0], lmax=config.lmax, lmax_mask=config.lmax)
                                    f_0_1_w = nmt.NmtField(compute_cls["mask"][0] * mask_in_maps, [T_map[1]], beam=bls_beam[0], lmax=config.lmax, lmax_mask=config.lmax)
                                    w00 = nmt.NmtWorkspace.from_fields(f_0_0_w, f_0_1_w, b_bin)
                                else:
                                    w00 = nmt.NmtWorkspace.from_fields(f_0_0, f_0_1, b_bin)
                            getattr(cls_out, attr).append((nmt.compute_full_master(f_0_0, f_0_1, b_bin, workspace=w00))[0])
                        f2_0 = nmt.NmtField(compute_cls["mask"][field_Q], [Q_map[0], U_map[0]], purify_b=compute_cls["nmt_purify_B"], purify_e=compute_cls["nmt_purify_E"], beam=beam_nmt, lmax=config.lmax, lmax_mask=config.lmax)
                        f2_1 = nmt.NmtField(compute_cls["mask"][field_Q], [Q_map[1], U_map[1]], purify_b=compute_cls["nmt_purify_B"], purify_e=compute_cls["nmt_purify_E"], beam=beam_nmt, lmax=config.lmax, lmax_mask=config.lmax)
                        if idx==0:
                            if mask_in_maps is not None:
#                                f2_0_w = nmt.NmtField(compute_cls["mask"][field_Q] * mask_in_maps, [Q_map[0], U_map[0]], purify_b=compute_cls["nmt_purify_B"], purify_e=compute_cls["nmt_purify_E"], beam=beam_nmt, lmax=config.lmax, lmax_mask=config.lmax)
#                                f2_1_w = nmt.NmtField(compute_cls["mask"][field_Q] * mask_in_maps, [Q_map[1], U_map[1]], purify_b=compute_cls["nmt_purify_B"], purify_e=compute_cls["nmt_purify_E"], beam=beam_nmt, lmax=config.lmax, lmax_mask=config.lmax)
                                pureE = "EBpurify" in compute_cls["path"] or "Epurify" in compute_cls["path"]
                                f2_0_w = nmt.NmtField(compute_cls["mask"][field_Q] * mask_in_maps, [Q_map[0], U_map[0]], purify_b= ("Bpurify" in compute_cls["path"]), purify_e=pureE, beam=beam_nmt, lmax=config.lmax, lmax_mask=config.lmax)
                                f2_1_w = nmt.NmtField(compute_cls["mask"][field_Q] * mask_in_maps, [Q_map[1], U_map[1]], purify_b= ("Bpurify" in compute_cls["path"]), purify_e=pureE, beam=beam_nmt, lmax=config.lmax, lmax_mask=config.lmax)
                                w22 = nmt.NmtWorkspace.from_fields(f2_0_w, f2_1_w, b_bin)
                            else:
                                w22 = nmt.NmtWorkspace.from_fields(f2_0, f2_1, b_bin)
                        if "EE" in config.field_cls_out:
                            getattr(cls_out, attr).append(w22.decouple_cell(nmt.compute_coupled_cell(f2_0, f2_1))[0])
                        if "BB" in config.field_cls_out:
                            getattr(cls_out, attr).append(w22.decouple_cell(nmt.compute_coupled_cell(f2_0, f2_1))[3])
                        if ("EETE" in config.field_cls_out or "BBTE" in config.field_cls_out) or ("BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out):
                            if idx==0:
                                if mask_in_maps is not None:
                                    w02 = nmt.NmtWorkspace.from_fields(f_0_0_w, f2_1_w, b_bin)
                                else:
                                    w02 = nmt.NmtWorkspace.from_fields(f_0_0, f2_1, b_bin)
                        if "EETE" in config.field_cls_out or "BBTE" in config.field_cls_out:
                            getattr(cls_out, attr).append(w02.decouple_cell(nmt.compute_coupled_cell(f_0_0, f2_1))[0])
                        if "BBEB" in config.field_cls_out or "BTEEB" in config.field_cls_out:
                            getattr(cls_out, attr).append(w22.decouple_cell(nmt.compute_coupled_cell(f2_0, f2_1))[1])
                        if "BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out:
                            getattr(cls_out, attr).append(w02.decouple_cell(nmt.compute_coupled_cell(f_0_0, f2_1))[1])

        else:
            if output_data.ndim == 1:
                if config.spectra_comp == 'anafast':
                    getattr(cls_out, attr).append(compute_anafast_scalar(output_data, compute_cls["mask"], bls_beam[0]))
                else:
                    getattr(cls_out, attr).append(compute_namaster_scalar(output_data, compute_cls["mask"], bls_beam[0], compute_cls["field_cls_in"]))
            else:
                if 'QU' not in compute_cls["field_cls_in"]:
                    for field in range(output_data.shape[0]):
                        if config.spectra_comp == 'anafast':
                            getattr(cls_out, attr).append(compute_anafast_scalar(
                                output_data[field], compute_cls["mask"][field], bls_beam[field]))
                        else:
                            getattr(cls_out, attr).append(compute_namaster_scalar(
                            output_data[field], compute_cls["mask"][field], bls_beam[field], compute_cls["field_cls_in"][field]))
                    
                    if "EETE" in config.field_cls_out or "BBTE" in config.field_cls_out:
                        if config.spectra_comp == 'anafast':
                            mask_ =  compute_cls["mask"][1] if np.mean(np.ceil(compute_cls["mask"][0])) > np.mean(np.ceil(compute_cls["mask"][1])) else compute_cls["mask"][0]
                            getattr(cls_out, attr).append(compute_anafast_cross_scalars(
                                output_data[0], output_data[1], mask_, bls_beam[0], bls_beam[1]))
                        elif config.spectra_comp == 'namaster':                        
                            getattr(cls_out, attr).append(compute_namaster_cross_scalars(
                                output_data[0], output_data[1], compute_cls["mask"][0], compute_cls["mask"][1],
                                bls_beam[0], bls_beam[1], compute_cls["field_cls_in"][0], compute_cls["field_cls_in"][1]
                            ))
                    
                    if "BBEB" in config.field_cls_out or "BTEEB" in config.field_cls_out:
                        field_E, field_B = (1, 2) if "T" in compute_cls["field_cls_in"] else (0, 1)
                        if config.spectra_comp == 'anafast':
                            mask_ =  compute_cls["mask"][field_B] if np.mean(np.ceil(compute_cls["mask"][field_E])) > np.mean(np.ceil(compute_cls["mask"][field_B])) else compute_cls["mask"][field_E]
                            getattr(cls_out, attr).append(
                                compute_anafast_cross_scalars(output_data[field_E], output_data[field_B], 
                                mask_, bls_beam[field_E], bls_beam[field_B]))
                        elif config.spectra_comp == 'namaster':
                            getattr(cls_out, attr).append(compute_namaster_cross_scalars(
                                output_data[field_E], output_data[field_B], compute_cls["mask"][field_E], compute_cls["mask"][field_B],
                                bls_beam[field_E], bls_beam[field_B], compute_cls["field_cls_in"][field_E], compute_cls["field_cls_in"][field_B]
                            ))

                    if "BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out:
                        field_B = 2 if "E" in compute_cls["field_cls_in"] else 1
                        if config.spectra_comp == 'anafast':
                            mask_ =  compute_cls["mask"][field_B] if np.mean(np.ceil(compute_cls["mask"][0])) > np.mean(np.ceil(compute_cls["mask"][field_B])) else compute_cls["mask"][0]
                            getattr(cls_out, attr).append(compute_anafast_cross_scalars(
                                output_data[0], output_data[field_B], mask_, bls_beam[0], bls_beam[field_B]))
                        elif config.spectra_comp == 'namaster':
                            getattr(cls_out, attr).append(compute_namaster_cross_scalars(
                                output_data[0], output_data[field_B], compute_cls["mask"][0], compute_cls["mask"][field_B],
                                bls_beam[0], bls_beam[field_B], compute_cls["field_cls_in"][0], compute_cls["field_cls_in"][field_B]
                            ))
                else:
                    if "T" in compute_cls["field_cls_in"]:
                        field_Q = 1
                        field_B = 1 if "EE" not in config.field_cls_out else 2 if "BB" in config.field_cls_out else None
                        field_E = 1 if "EE" in config.field_cls_out else None
                        beam_nmt = bls_beam[0]
                        T_map, Q_map, U_map = output_data[0], output_data[1], output_data[2]
                    else:
                        mask_ =  compute_cls["mask"][0]
                        field_Q = 0
                        field_B = 0 if "EE" not in config.field_cls_out else 1 if "BB" in config.field_cls_out else None
                        field_E = 0 if "EE" in config.field_cls_out else None
                        beam_nmt = hp.gauss_beam(np.radians(config.fwhm_out/60.), lmax=config.lmax, pol=False)
                        if config.pixel_window_out:
                            beam_nmt *= hp.pixwin(config.nside, lmax=config.lmax, pol=False)
                        T_map, Q_map, U_map = np.zeros_like(output_data[0]), output_data[0], output_data[1]

                    get_cross = any(x in config.field_cls_out for x in ["EETE", "BBTE", "BBEB", "BTEEB", "BBTB", "EBTB"])
                    if config.spectra_comp == 'anafast':
                        cls_s2 = compute_anafast_full_TQU(
                            T_map, Q_map, U_map,
                            compute_cls["mask"][0], compute_cls["mask"][field_Q],
                            bls_beam[0], bls_beam[field_E], bls_beam[field_B], get_cross=get_cross
                        )
                        if "TT" in config.field_cls_out:
                            getattr(cls_out, attr).append(cls_s2[0])
                        if "EE" in config.field_cls_out:
                            getattr(cls_out, attr).append(cls_s2[1])
                        if "BB" in config.field_cls_out:
                            getattr(cls_out, attr).append(cls_s2[2])
                        if get_cross:
                            if "EETE" in config.field_cls_out or "BBTE" in config.field_cls_out:
                                getattr(cls_out, attr).append(cls_s2[3])
                            if "BBEB" in config.field_cls_out or "BTEEB" in config.field_cls_out:
                                getattr(cls_out, attr).append(cls_s2[4])
                            if "BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out:
                                getattr(cls_out, attr).append(cls_s2[5])
                    elif config.spectra_comp == 'namaster':
                        if "TT" in config.field_cls_out:
                            f_0 = nmt.NmtField(compute_cls["mask"][0], [T_map], beam=bls_beam[0],lmax=config.lmax,lmax_mask=config.lmax)
                            if idx==0:
                                if mask_in_maps is not None:
                                    f_0_w = nmt.NmtField(compute_cls["mask"][0] * mask_in_maps, [T_map], beam=bls_beam[0], lmax=config.lmax, lmax_mask=config.lmax)
                                    w00 = nmt.NmtWorkspace.from_fields(f_0_w, f_0_w, b_bin)
                                else:
                                    w00 = nmt.NmtWorkspace.from_fields(f_0, f_0, b_bin)
                            getattr(cls_out, attr).append((nmt.compute_full_master(f_0, f_0, b_bin, workspace=w00))[0])
                        f2 = nmt.NmtField(compute_cls["mask"][field_Q], [Q_map, U_map], purify_b=compute_cls["nmt_purify_B"], purify_e=compute_cls["nmt_purify_E"], beam=beam_nmt, lmax=config.lmax, lmax_mask=config.lmax)
                        if idx==0:
                            if mask_in_maps is not None:
#                                f2_w = nmt.NmtField(compute_cls["mask"][field_Q] * mask_in_maps, [Q_map, U_map], purify_b=compute_cls["nmt_purify_B"], purify_e=compute_cls["nmt_purify_E"], beam=beam_nmt, lmax=config.lmax, lmax_mask=config.lmax)
                                pureE = "EBpurify" in compute_cls["path"] or "Epurify" in compute_cls["path"]
                                f2_w = nmt.NmtField(compute_cls["mask"][field_Q] * mask_in_maps, [Q_map, U_map], purify_b= ("Bpurify" in compute_cls["path"]), purify_e=pureE, beam=beam_nmt, lmax=config.lmax, lmax_mask=config.lmax)
                                w22 = nmt.NmtWorkspace.from_fields(f2_w, f2_w, b_bin)
                            else:
                                w22 = nmt.NmtWorkspace.from_fields(f2, f2, b_bin)
                        if "EE" in config.field_cls_out:
                            getattr(cls_out, attr).append(w22.decouple_cell(nmt.compute_coupled_cell(f2, f2))[0])
                        if "BB" in config.field_cls_out:
                            getattr(cls_out, attr).append(w22.decouple_cell(nmt.compute_coupled_cell(f2, f2))[3])
                        if ("EETE" in config.field_cls_out or "BBTE" in config.field_cls_out) or ("BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out):
                            if idx==0:
                                if mask_in_maps is not None:
                                    w02 = nmt.NmtWorkspace.from_fields(f_0_w, f2_w, b_bin)
                                else:
                                    w02 = nmt.NmtWorkspace.from_fields(f_0, f2, b_bin)
                        if "EETE" in config.field_cls_out or "BBTE" in config.field_cls_out:
                            getattr(cls_out, attr).append(w02.decouple_cell(nmt.compute_coupled_cell(f_0, f2))[0])
                        if "BBEB" in config.field_cls_out or "BTEEB" in config.field_cls_out:
                            getattr(cls_out, attr).append(w22.decouple_cell(nmt.compute_coupled_cell(f2, f2))[1])
                        if "BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out:
                            getattr(cls_out, attr).append(w02.decouple_cell(nmt.compute_coupled_cell(f_0, f2))[1])

    for attr in vars(cls_out).keys():
        setattr(cls_out, attr, np.array(getattr(cls_out, attr)))

    return cls_out
            
def get_bls(nside: int, fwhm: float, lmax: int, field_cls_out: str, pixel_window_out: bool = True) -> np.ndarray:
    """
    Get the beam transfer functions for the required fields and lmax.

    Parameters
    ----------
        nside : int
            HEALPix nside parameter.
        fwhm : float
            Full width at half maximum of the beam in arcminutes.
        lmax : int
            Maximum multipole to consider.
        field_cls_out : str
            Fields for which to compute the beam transfer functions.
        pixel_window_out : bool, optional
            Whether to consider the pixel window function in the beam transfer functions. Default is True.
    
    Returns
    -------
        bls_beam : np.ndarray
            Beam transfer functions for the specified fields and lmax.
    """

    bls_beam = []

    bl_ = hp.gauss_beam(np.radians(fwhm/60.), lmax=lmax, pol=True).T
    if pixel_window_out:
        pw_ = hp.pixwin(nside, lmax=lmax, pol=True)
        bl_[0] *= pw_[0]
        bl_[1] *= pw_[1]
        bl_[2] *= pw_[1]

    if 'TT' in field_cls_out:
        bls_beam.append(bl_[0])
    if 'EE' in field_cls_out:
        bls_beam.append(bl_[1])
    if 'BB' in field_cls_out:
        bls_beam.append(bl_[2])

    bls_beam = np.array(bls_beam)
    if bls_beam.ndim == 1:
        bls_beam = bls_beam[np.newaxis, :]

    return bls_beam

def _get_fields_in_for_cls(field_out,field_cls_out):
    """
    Get the fields to be used for computing the spectra based on the loaded compsep outputs and the desired angular power spectra.

    Parameters
    ----------
        field_out : str
            Fields loaded from the component separation outputs. It can be one of the following:
            - 'T', 'E', 'B': Single fields for temperature, E-mode polarization, and B-mode polarization.
            - 'QU', 'QU_E', 'QU_B': Combined fields for Q and U polarization.
            - 'TQU': Combined fields for temperature, Q, and U polarization.
            - 'TEB': Combined fields for temperature, E-mode, and B-mode polarization.
        field_cls_out : str
            Fields for which to compute the angular power spectra. 
    
    Returns
    -------
        str
            Fields to be used for computing the angular power spectra based on the loaded compsep outputs and the desired angular power spectra.
    """

    if field_out == "TQU":
        if field_cls_out in ["EE", "BB", "EEBB", "EEBBEB"]:
            return "QU"
        elif field_cls_out == "TT":
            return "T"
        else:
            return "TQU"
    elif field_out in ['T', 'E', 'B', 'QU', 'QU_E', 'QU_B']:
        return field_out
    elif field_out in ["TEB", "EB"]:
        field_cls_in = ""
        if ("TT" in field_cls_out) and (field_out == "TEB"):
            field_cls_in += "T"
        if "EE" in field_cls_out:
            field_cls_in += "E"
        if "BB" in field_cls_out:
            field_cls_in += "B"
        return field_cls_in

def _check_fields_for_cls(field_out: str, field_cls_out: str) -> None:
    """
    Check if the fields returned by component separation are compatible with the desired angular power spectra.

    Parameters
    ----------
        field_out : str
            Fields loaded from the component separation outputs. It can be one of the following:
            - 'T', 'E', 'B', 'QU', 'QU_E', 'QU_B', 'TQU', 'TEB'
        field_cls_out : str
            Fields for which to compute the angular power spectra. It can be one of the following:
            - 'TT', 'EE', 'BB', 'TTEE', 'TTEETE', 'TTBB', 'TTBBTB', 'EEBB', 'EEBBEB', 'TTEEBB', 'TTEEBBTEEBTB'
    
    Raises
    ------
        ValueError
            If the fields are not compatible.
    """
    valid_fields = ['T', 'E', 'B', 'QU', 'QU_E', 'QU_B', 'EB', 'TQU', 'TEB']
    valid_cls_out = [
        'TT', 'EE', 'BB',
        'TTEE', 'TTEETE',
        'TTBB', 'TTBBTB',
        'EEBB', 'EEBBEB',
        'TTEEBB', 'TTEEBBTEEBTB'
    ]

    if field_out not in valid_fields:
        raise ValueError(f'Invalid field_out: "{field_out}". Must be one of {valid_fields}.')
    if field_cls_out not in valid_cls_out:
        raise ValueError(f'Invalid field_cls_out: "{field_cls_out}". Must be one of {valid_cls_out}.')

    if field_out in ['T','E','B'] and field_cls_out != f"{field_out}{field_out}":
        raise ValueError(f'If field_out is "{field_out}", field_cls_out must be "{field_out}{field_out}".')

    if field_out in ["QU", "EB"] and field_cls_out not in ["EE", "BB", "EEBB", "EEBBEB"]:
        raise ValueError(f'If field_out is "{field_out}", field_cls_out must be one of ["EE", "BB", "EEBB", "EEBBEB"].')

    if field_out == "QU_E" and field_cls_out != "EE":
        raise ValueError('If field_out is "QU_E", field_cls_out must be "EE".')
    if field_out == "QU_B" and field_cls_out != "BB":
        raise ValueError('If field_out is "QU_B", field_cls_out must be "BB".')

    if field_out in ["TQU", "TEB"] and field_cls_out not in valid_cls_out:
        raise ValueError(f'Invalid combination for field_out="{field_out}" and field_cls_out="{field_cls_out}".')

def _load_cls(path: str, components:list, field_cls_out: str, mask_folder: str,
    nside: int, lmax: int, fwhm_out: float, nsim: Optional[str] = None, return_Dell: bool = False) -> np.ndarray:
    """
    Load angular power spectra of requested component separation outputs computed according to mask_type.

    Parameters
    ----------
        path : str
            Path to the directory containing the power spectra
        components : list or str
            List of component names or a single component name for which spectra have to be loaded.
        field_cls_out : str
            The field(s) stored in the files to load.
        mask_folder: str
            Folder containing information about the masking approach used to compute angular power spectra.
        nside : int
            HEALPix resolution associated to the component separation outputs on which power spectrum has been computed.
        lmax : int
            Maximum multipole used to compute angular power spectra.
        fwhm_out : float
            Full width at half maximum of the output maps in arcminutes on which power spectrum has been computed.
        nsim : int or str, optional
            Simulation index of the compsep outputs. If None, it will look for files without nsim label. Default: None.
        return_Dell: bool
            If True, it will look for 'Dls' instead of 'Cls'. Default: False.

    Returns
    -------
        np.ndarray
            Loaded computed angular power spectra for the requested component separation run and output components.
    
    """
    nsim = _format_nsim(nsim)

    loaded_cls = []

    if isinstance(components, str):
        components = [components]
    if not isinstance(components, list):
        raise ValueError("components must be a string or a list of strings.")

    pre_filename = "Dls" if return_Dell else "Cls"

    for component in components:
        component_name = component.split('/')[0] if '/' in component else component
        filename = os.path.join(
            path,
            f"spectra/{mask_folder}/{component}/{pre_filename}_{field_cls_out}_{component_name}_{fwhm_out}acm_ns{nside}_lmax{lmax}"
        )
        if nsim is not None:
            filename += f"_{nsim}.fits"
        loaded_cls.append(hp.read_cl(filename))

    return np.array(loaded_cls)

class BBin:
    """Class for binning power spectra into bands of ell values."""

    def __init__(self, ell_bins, lmax, is_Dell=False, w_ell=None):
        """
        ell_bins : list of arrays
            Each element contains the ell values in that band.
        is_Dell : bool
            If True, bin Dell instead of Cl.
        """
        self.ell_bins = ell_bins
        self.n_bands = len(ell_bins)
        self.lmax = lmax
        self.is_Dell = is_Dell
        if w_ell is not None:
            self.w_ell = w_ell
        else:
            self.w_ell = np.ones(lmax + 1)

    # --------------------------
    # Constructors
    # --------------------------

    @classmethod
    def from_edges(bin_class, ell_ini, ell_end, is_Dell=False):
        """
        Create bins from lower/upper edges.
        
        ell_ini, ell_end : arrays of same length
            Each pair defines one band: ell_ini[i] <= ell < ell_end[i]
        """
        ell_bins = []
        for lo, hi in zip(ell_ini, ell_end):
            ell_bins.append(np.arange(lo, hi))
        return bin_class(ell_bins, lmax=ell_end[-1]-1, is_Dell=is_Dell)

    @classmethod
    def from_lmax_linear(bin_class, lmax, nlb, is_Dell=False):
        """
        Create linear bins of width nlb up to lmax.
        
        lmax : int
        nlb  : bin width (delta_ell)
        """
        ell_bins = []
        for lo in range(2, lmax + 1, nlb):
            hi = min(lo + nlb - 1, lmax)
            ell_bins.append(np.arange(lo, hi + 1))
        return bin_class(ell_bins, lmax=lmax, is_Dell=is_Dell)

    # --------------------------
    # Binning function
    # --------------------------

    def bin_cell(self, cl):
        """
        Bin power spectra.

        Accepts:
        - cl shape (lmax+1,)
        - cl shape (n_fields, lmax+1)

        Returns:
        - if input was 1D: shape (n_bands,)
        - if input was 2D: shape (n_fields, n_bands)
        """
        cl = np.asarray(cl)

        was_1d = (cl.ndim == 1)
        if was_1d:
            cl = cl[None, :]  # -> (1, lmax+1)
        elif cl.ndim != 2:
            raise ValueError(f"cl must be 1D or 2D, got shape {cl.shape}")

        n_fields, lmax_plus_1 = cl.shape
        cl_binned = np.zeros((n_fields, self.n_bands), dtype=cl.dtype)

        for ib, ell_vals in enumerate(self.ell_bins):
            ell_vals = np.asarray(ell_vals)
            ell_vals = ell_vals[ell_vals < lmax_plus_1]  # safety

            if ell_vals.size == 0:
                continue

            if self.is_Dell:
                factor = ell_vals * (ell_vals + 1) / (2.0 * np.pi)  # (n_ell,)
                band_values = cl[:, ell_vals] * factor[None, :]     # broadcast
            else:
                band_values = cl[:, ell_vals]

            cl_binned[:, ib] = np.sum(band_values * self.w_ell[ell_vals][None, :], axis=1) / np.sum(self.w_ell[ell_vals])

        return cl_binned[0] if was_1d else cl_binned

    def get_effective_ells(self):
        """
        Get effective ell values for each band.

        Returns:
        - effective_ells: array of shape (n_bands,)
        """
        effective_ells = np.zeros(self.n_bands)
        for ib, ell_vals in enumerate(self.ell_bins):
            ell_vals = np.asarray(ell_vals)
            effective_ells[ib] = np.sum(ell_vals * self.w_ell[ell_vals]) / np.sum(self.w_ell[ell_vals])
        return effective_ells

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]

