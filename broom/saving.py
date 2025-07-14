import os
import re
import numpy as np
import healpy as hp
from typing import Any, Optional, Union, Dict
from .routines import _log
from .configurations import Configs
from types import SimpleNamespace
import fnmatch
import sys

def _save_compsep_products(
    config: Configs,
    output_maps: SimpleNamespace,
    compsep_run: Dict[str, Any],
    nsim: Optional[str] = None
) -> None:
    """
    Save component separation products to disk based on the method and simulation parameters.

    Parameters
    ------------
        config: Configs
            Configuration object. It contains paths and parameters for saving outputs.
        output_maps: SimpleNamespace 
            Object containing separated map outputs as attributes.
        compsep_run: Dict 
            Dictionary describing the component separation method and setup.
        nsim: (int, optional)
            Simulation index for saving multiple realizations.

    Returns
    ------------
        None
            It saves the output maps to disk in the specified directory structure 
            based on the component separation method and configuration.
    """

    #if 'path_out' not in compsep_run:
    compsep_run["path_out"] = _get_full_path_out(config, compsep_run)

    if compsep_run["method"] in ["cilc", "c_ilc", "mc_cilc","cpilc", "c_pilc"]:
        if 'mixed' in compsep_run["path_out"]:
            with open(os.path.join(compsep_run["path_out"], "constraints_info.txt"), "w") as f: 
                f.write("# Moments and deprojection constraints used in the component separation run\n\n")
                
                moments_header = "# Moments per needlet band (if applicable):\n"
                f.write(moments_header)
                if compsep_run["domain"] == "needlet":
                    for row in compsep_run["constraints"]["moments"]:
                        f.write(" ".join(row) + "\n")
                else:
                    f.write(" ".join(compsep_run["constraints"]["moments"]) + "\n")
                f.write("\n")
                
                depro_header = "# Deprojection coefficients per needlet band (if applicable):\n"
                f.write(depro_header)
                if compsep_run["domain"] == "needlet":
                    for row in compsep_run["constraints"]["deprojection"]:
                        f.write(" ".join(map(str, row)) + "\n")
                else:
                    f.write(" ".join(map(str, compsep_run["constraints"]["deprojection"])) + "\n")
                f.write("\n")
                
                beta_d_header = "# Beta_d values per needlet band (if applicable):\n"
                f.write(beta_d_header)
                if compsep_run["domain"] == "needlet":
                    for row in compsep_run["constraints"]["beta_d"]:
                        f.write(f"{row}\n")
                else:
                    f.write(f'{compsep_run["constraints"]["beta_d"]}\n')
                f.write("\n")
                
                T_d_header = "# T_d values per needlet band (if applicable):\n"
                f.write(T_d_header)
                if compsep_run["domain"] == "needlet":
                    for row in compsep_run["constraints"]["T_d"]:
                        f.write(f"{row}\n")
                else:
                    f.write(f'{compsep_run["constraints"]["T_d"]}\n')
                f.write("\n")
                
                beta_s_header = "# Beta_s values per needlet band (if applicable):\n"
                f.write(beta_s_header)
                if compsep_run["domain"] == "needlet":
                    for row in compsep_run["constraints"]["beta_s"]:
                        f.write(f"{row}\n")
                else:
                    f.write(f'{compsep_run["constraints"]["beta_s"]}\n')
                f.write("\n")

    for attr_name, attr_values in vars(output_maps).items():
        if attr_name == "total":
            label_out = "output_total"
        elif attr_name == "cmb":
            label_out = "output_cmb"
        elif attr_name == "m":
            label_out = "fgd_complexity"
        else:
            label_out = f"{attr_name}_residuals"

        path_c = os.path.join(compsep_run["path_out"], f"{label_out}")
        os.makedirs(path_c, exist_ok=True)

        if compsep_run["method"] in ["gilc","gpilc"]:
            if nsim is not None:
                path_c = os.path.join(path_c, f"{nsim}")
                os.makedirs(path_c, exist_ok=True)

            for f, freq in enumerate(compsep_run["channels_out"]):
                tag = config.instrument.channels_tags[freq]
                filename = (
                    f"{config.field_out}_{label_out}_{tag}_{config.fwhm_out}acm_"
                    f"ns{config.nside}_lmax{config.lmax}"
                )
                if nsim is not None:
                    filename += f"_{nsim}"
                filename += ".fits"

                hp.write_map(os.path.join(path_c, filename), attr_values[f], overwrite=True)

        elif (
            compsep_run["method"] in ["fgd_diagnostic", "fgd_P_diagnostic"]
            and compsep_run["domain"] == "needlet"
        ):
            if nsim is not None:
                path_c = os.path.join(path_c, f"{nsim}")
                os.makedirs(path_c, exist_ok=True)
            
            for j in range(attr_values.shape[-2]):
                filename = (
                    f"{config.field_out}_{label_out}_nl{j}_{config.fwhm_out}acm_"
                    f"ns{config.nside}_lmax{config.lmax}"
                )
                if nsim is not None:
                    filename += f"_{nsim}"
                filename += ".fits"
                if attr_values.ndim == 2:
                    hp.write_map(os.path.join(path_c, filename), attr_values[j], overwrite=True)
                elif attr_values.ndim == 3:
                    hp.write_map(os.path.join(path_c, filename), attr_values[:,j], overwrite=True)
        else:              
            filename = (
                f"{config.field_out}_{label_out}_{config.fwhm_out}acm_"
                f"ns{config.nside}_lmax{config.lmax}"
            )
            if nsim is not None:
                filename += f"_{nsim}"
            filename += ".fits"
            hp.write_map(os.path.join(path_c, filename), attr_values, overwrite=True)
            
def _save_residuals_template(
    config: Configs,
    output_maps: SimpleNamespace,
    compsep_run: Dict[str, Any],
    nsim: Optional[str] = None
) -> None:
    """
    Save residual foreground templates.

    Parameters
    -----------
        config: Configs
            Configuration object. It contains paths and parameters for saving outputs.
        output_maps: SimpleNamespace
            Object containing separated map outputs as attributes.
        compsep_run: Dict
            Dictionary describing the component separation method and setup.
        nsim: str, optional
            Simulation index for saving multiple realizations.

    Returns
    -----------
        None
            It saves the fgd residuals estimate maps to disk in the specified directory structure
            based on the component separation method and configuration.
    
    """
    path_out = os.path.join(config.path_outputs, compsep_run["compsep_path"])

    gnilc_run = (re.search(r'(gilc_[^/]+)', compsep_run["gnilc_path"])).group(1)
    if "needlet" in gnilc_run:
        folder_after = (compsep_run["gnilc_path"]).split(gnilc_run + "/")[1].split("/")[0]
        gnilc_run += f"_{folder_after}"

    for attr_name, attr_values in vars(output_maps).items():
        if attr_name == "total":
            label_out = "fgres_templates"
        elif attr_name == "fgds":
            label_out = "fgres_templates_ideal"
        else:
            label_out = f"fgres_templates_{attr_name}"

        path_c = os.path.join(path_out, f"{label_out}", gnilc_run)
        os.makedirs(path_c, exist_ok=True)

        filename = (
            f"{config.field_out}_{label_out}_{config.fwhm_out}acm_"
            f"ns{config.nside}_lmax{config.lmax}"
        )
        if nsim is not None:
            filename += f"_{nsim}"
        filename += ".fits"

        hp.write_map(os.path.join(path_c, filename), attr_values, overwrite=True)

def _get_full_path_out(config: Configs, compsep_run: Dict[str, Any]) -> str:
    """
    Constructs the full output path for component separation products based on configuration and run options.

    Parameters
    -----------
        config: Configs
            Configuration object.
        compsep_run: dict
            Dictionary containing method and domain setup.

    Returns
    --------
        str
            Full path where outputs should be saved.
    """
    
    if compsep_run["method"] in ["mc_ilc", "mc_cilc", "c_ilc", "c_pilc"]:
        complete_path = f'{compsep_run["method"]}_{compsep_run["domain"]}_bias{compsep_run["ilc_bias"]}_nls{"-".join(map(str, compsep_run["special_nls"]))}' 
    elif compsep_run["method"] == "mcilc":
        complete_path = f'{compsep_run["method"]}_{compsep_run["domain"]}'
    elif compsep_run["method"] in ["gilc", "gpilc"]:
        complete_path = f'{compsep_run["method"]}_{compsep_run["domain"]}_bias{compsep_run["ilc_bias"]}'
        if compsep_run["domain"] == "pixel":
            if compsep_run["m_bias"] != 0:
                complete_path += f"_m{compsep_run['m_bias']:+}"
            if compsep_run["depro_cmb"] is not None:
                complete_path += f"_deproCMB{compsep_run['depro_cmb']}"
        elif compsep_run["domain"] == "needlet":
            m_bias_array = np.array(compsep_run["m_bias"])
            if any(m_bias_array != 0):
                for m_bias in np.unique(m_bias_array[m_bias_array != 0]):
                    nls_bias = np.where(m_bias_array == m_bias)[0]
                    complete_path += f"_m{m_bias:+}_nls{'-'.join(map(str, nls_bias))}"

            depro_array = np.array(compsep_run["depro_cmb"])        
            if any(depro_array != None):
                for depro_val in np.unique(depro_array[depro_array != None]):
                    nls_depro = np.where(depro_array == depro_val)[0]
                    complete_path += f"_deproCMB{depro_val}_nls{'-'.join(map(str, nls_depro))}"
    else:
        complete_path = f'{compsep_run["method"]}_{compsep_run["domain"]}_bias{compsep_run["ilc_bias"]}'

    if compsep_run["method"] in ["gilc", "gpilc", "fgd_diagnostic", "fgd_P_diagnostic"]:
        if not compsep_run["cmb_nuisance"]:
            complete_path += "_nocmbnuisance"

    if compsep_run["method"] != "mcilc":
        if compsep_run["domain"] == "pixel":
            if compsep_run["cov_noise_debias"] != 0.:
                complete_path += f"_noidebias{compsep_run['cov_noise_debias']}"
        elif compsep_run["domain"] == "needlet":
            debias_array = np.array(compsep_run["cov_noise_debias"])        
            if any(debias_array != 0.):
                for debias_val in np.unique(debias_array[debias_array != 0.]):
                    nls_debias = np.where(debias_array == debias_val)[0]
                    complete_path += f"_noidebias{debias_val}_nls{'-'.join(map(str, nls_debias))}"

#    if (config.leakage_correction is not None) and ("QU" in config.field_in) and (config.mask_type == "observed_patch"):
    if (config.leakage_correction is not None) and ("QU" in config.field_in) and (config.mask_observations is not None):
        leak_def = (config.leakage_correction).split("_")[0] + (config.leakage_correction).split("_")[1] 
        if "_recycling" in config.leakage_correction:
            if "_iterations" in config.leakage_correction:
                iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
                leak_def += f'_iters{iterations}'
        complete_path += f"_{leak_def}"

    if compsep_run["method"] in ["cilc", "c_ilc", "mc_cilc","cpilc", "c_pilc"]:
        if compsep_run["domain"] == "pixel":
            mom_text = "".join(compsep_run["constraints"]["moments"])
        elif compsep_run["domain"] == "needlet":
            if all(list(set(row)) == list(set(compsep_run["constraints"]["moments"][0])) for row in compsep_run["constraints"]["moments"]):
                mom_text = "".join(compsep_run["constraints"]["moments"][0])
            else:
                mom_text = ""
                for idx, row in enumerate(compsep_run["constraints"]["moments"]):
                    if idx == 0:
                        mom_text += "".join(row)
                    else:
                        if list(set(row)) != list(set(compsep_run["constraints"]["moments"][idx-1])):
                            mom_text += "_" + "".join(row)
        
        if isinstance(compsep_run["constraints"]["deprojection"], float):  # Handle case where deprojection is a single float
            all_depros = [compsep_run["constraints"]["deprojection"]]
        elif isinstance(compsep_run["constraints"]["deprojection"], list):  # Handle case where deprojection is a list
            if all(isinstance(sublist, list) for sublist in compsep_run["constraints"]["deprojection"]):  # Check if it's a list of lists
                all_depros = list(set(element for sublist in compsep_run["constraints"]["deprojection"] for element in sublist))
            else:  # Handle case where deprojection is a flat list
                all_depros = list(set(compsep_run["constraints"]["deprojection"]))
        if len(all_depros)==1:
            if all_depros[0] != 0.:
                mom_text += f"_depro{all_depros[0]}"
        else:
            mom_text += f"_mixeddepro"
            
        if isinstance(compsep_run["constraints"]["beta_d"], float):  # Handle case where beta_d is a single float
            all_betad = [compsep_run["constraints"]["beta_d"]]
        elif isinstance(compsep_run["constraints"]["beta_d"], list):  # Handle case where beta_d is a list
            if all(isinstance(sublist, list) for sublist in compsep_run["constraints"]["beta_d"]):  # Check if it's a list of lists
                all_betad = list(set(element for sublist in compsep_run["constraints"]["beta_d"] for element in sublist))
            else:  # Handle case where beta_d is a flat list
                all_betad = list(set(compsep_run["constraints"]["beta_d"]))
        if len(all_betad)==1:
            if all_betad[0] != 1.54:
                mom_text += f"_bd{all_betad[0]}"
        else:
            mom_text += f"_mixedbd"

        if isinstance(compsep_run["constraints"]["T_d"], float):  # Handle case where T_d is a single float
            all_Td = [compsep_run["constraints"]["T_d"]]
        elif isinstance(compsep_run["constraints"]["T_d"], list):  # Handle case where T_d is a list
            if all(isinstance(sublist, list) for sublist in compsep_run["constraints"]["T_d"]):  # Check if it's a list of lists
                all_Td = list(set(element for sublist in compsep_run["constraints"]["T_d"] for element in sublist))
            else:  # Handle case where T_d is a flat list
                all_Td = list(set(compsep_run["constraints"]["T_d"]))
        if len(all_Td)==1:
            if all_Td[0] != 20.:
                mom_text += f"_Td{all_Td[0]}"
        else:
            mom_text += f"_mixedTd"

        if isinstance(compsep_run["constraints"]["beta_s"], float):  # Handle case where beta_s is a single float
            all_betas = [compsep_run["constraints"]["beta_s"]]
        elif isinstance(compsep_run["constraints"]["beta_s"], list):  # Handle case where beta_s is a list
            if all(isinstance(sublist, list) for sublist in compsep_run["constraints"]["beta_s"]):  # Check if it's a list of lists
                all_betas = list(set(element for sublist in compsep_run["constraints"]["beta_s"] for element in sublist))
            else:  # Handle case where beta_s is a flat list
                all_betas = list(set(compsep_run["constraints"]["beta_s"]))
        if len(all_betas)==1:
            if all_betas[0] != -3.:
                mom_text += f"_bs{all_betas[0]}"
        else:
            mom_text += f"_mixedbs"

        if len(all_betas) > 1 or len(all_depros) > 1 or len(all_Td) > 1 or len(all_betad) > 1:
            case = 1
            while os.path.exists(os.path.join(config.path_outputs, complete_path, f"{mom_text}_case{case}")):
                case += 1
            mom_text += f"_case{case}"

        complete_path = os.path.join(complete_path, mom_text)

    if compsep_run["domain"] == "needlet":
        text_ = f"{compsep_run['needlet_config']['needlet_windows']}"
        if compsep_run["needlet_config"]["needlet_windows"] != "cosine":
            text_ += f'_B{compsep_run["needlet_config"]["width"]}'
            if compsep_run["needlet_config"]["merging_needlets"]:
                merging_needlets = compsep_run["needlet_config"]["merging_needlets"]
                if merging_needlets[0] != 0:
                    merging_needlets.insert(0,0)
                for j_low, j_high in zip(merging_needlets[:-1], merging_needlets[1:]):
                    text_ += f"_j{j_low}j{j_high-1}"
        else:
            for bandpeak in compsep_run["needlet_config"]["ell_peaks"]:
                text_ += f"_{bandpeak}"
        if compsep_run["b_squared"] or compsep_run["method"] in ["pilc", "cpilc", "c_pilc", "gpilc", "fgd_P_diagnostic"]:
            text_ += "_nlsquared"
        complete_path = os.path.join(complete_path, text_)

    if compsep_run["method"] in ["mcilc","mc_ilc","mc_cilc"]:
        text_ = compsep_run["mc_type"]
        for freq_tracer in compsep_run["channels_tracers"]:
            text_ += f"_{config.instrument.channels_tags[freq_tracer]}"
        text_ += f"_{compsep_run['n_patches']}patches"
        complete_path = os.path.join(complete_path, text_)

    path_out = os.path.join(config.path_outputs, complete_path)

    return path_out

def get_gnilc_maps(
    config: Configs,
    path_gnilc: str,
    field_in: Optional[str] = None,
    nsim: Optional[str] = None
) -> SimpleNamespace:
    """
    Load GNILC component separation results (total signal, noise residuals, and foreground residuals) of all frequency channels
    provided in the instrument object and for a given simulation run and field. 

    Parameters
    -----------
        config : Configs
            Configuration object containing instrument and output specifications.
        path_gnilc : str
            Root path to the GNILC output directory. The full path will be given by '{config.path_outputs}/{path_gnilc}'.
        field_in : Optional[str], default=None
            Type of field to load ("T", "QU", "EB", "TQU", "TEB", etc.). If None, default is config.field_out.
        nsim : Optional[Union[str, int]], default=None
            Simulation identifier, if any (used to select specific simulation output files).

    Returns
    --------
        gnilc_maps : SimpleNamespace
            A container with the following attributes:
            - total: np.ndarray of GNILC total signal maps.
            - noise: np.ndarray of noise residual maps (if available).
            - fgds: np.ndarray of foreground residual maps (if available).
    """
    if not os.path.exists(os.path.join(config.path_outputs, path_gnilc)):
        raise ValueError(f"Path {os.path.join(config.path_outputs, path_gnilc)} does not exist.")
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

    filepath = os.path.join(config.path_outputs, path_gnilc, "output_total")
    if nsim is not None:
        filepath = os.path.join(filepath, nsim)
    if not os.path.exists(filepath):
        raise ValueError(f"Path {filepath} does not contain the expected multifrequency maps.")

    setattr(gnilc_maps, "total", [])

    for f, freq in enumerate(config.instrument.frequency):
        tag = config.instrument.channels_tags[f]
        filename = f"{field_in}_output_total_{tag}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
        if nsim is not None:
            filename += f"_{nsim}"
        filename += ".fits"
        getattr(gnilc_maps, "total").append(hp.read_map(os.path.join(filepath, filename), field=gnilc_fields))
    setattr(gnilc_maps, "total", np.array(getattr(gnilc_maps, "total")))

    filepath = os.path.join(config.path_outputs, path_gnilc, "noise_residuals")
    if nsim is not None:
        filepath = os.path.join(filepath, nsim)
    if not os.path.exists(filepath):
        print(f"Warning: Path {filepath} does not contain the expected noise residuals. Noise debias will not be possible")
    else:
        setattr(gnilc_maps, "noise", [])
        for f, freq in enumerate(config.instrument.frequency):
            tag = config.instrument.channels_tags[f]
            filename = f"{field_in}_noise_residuals_{tag}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
            if nsim is not None:
                filename += f"_{nsim}"
            filename += ".fits"
            getattr(gnilc_maps, "noise").append(hp.read_map(os.path.join(filepath, filename), field=gnilc_fields))
        setattr(gnilc_maps, "noise", np.array(getattr(gnilc_maps, "noise")))

    filepath = os.path.join(config.path_outputs, path_gnilc, "fgds_residuals")
    if nsim is not None:
        filepath = os.path.join(filepath, nsim)
    if os.path.exists(filepath):
        _log(f"Path {filepath} contains the expected foregrounds residuals. The ideal template of foregrounds residuals with no CMB and noise contamination will be computed", verbose=config.verbose)
        setattr(gnilc_maps, "fgds", [])
        for f, freq in enumerate(config.instrument.frequency):
            tag = config.instrument.channels_tags[f]
            filename = f"{field_in}_fgds_residuals_{tag}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
            if nsim is not None:
                filename += f"_{nsim}"
            filename += ".fits"
            getattr(gnilc_maps, "fgds").append(hp.read_map(os.path.join(filepath, filename), field=gnilc_fields))
        setattr(gnilc_maps, "fgds", np.array(getattr(gnilc_maps, "fgds")))

    return gnilc_maps

def save_ilc_weights(
    config: Configs,
    w: np.ndarray,
    compsep_run: Dict,
    nside_: int,
    nl_scale: Optional[Union[int, None]] = None
) -> None:
    """
    Save ILC component separation weights to disk with appropriate metadata in filename.

    Parameters
    ----------
        config : Configs
            Configuration object.
        w : np.ndarray
            component separation weights to be saved.
        compsep_run : dict
            Dictionary with component separation parameters.
        nside_ : int
            HEALPix NSIDE resolution of the output.
        nl_scale : int, optional
            Needlet scale index for the corresponding ILC run.
    
    Returns
    -------
        None
            It saves the weights to disk in the specified directory structure 
            based on the component separation method and configuration.
    """
    path_w = os.path.join(compsep_run["path_out"], "weights")
    os.makedirs(path_w, exist_ok=True)
    filename = os.path.join(path_w, f"weights_{compsep_run['field']}_{config.fwhm_out}acm_ns{nside_}_lmax{config.lmax}")
    if nl_scale is not None:
        filename += f"_nl{nl_scale}"
    if compsep_run["nsim"] is not None:
        filename += f"_{compsep_run['nsim']}"
    np.save(filename, w)

def save_patches(
    config: Configs,
    patches: np.ndarray,
    compsep_run: Dict,
    nl_scale: Optional[Union[int, None]] = None
) -> None:
    """
    Save MC-ILC patches to disk with appropriate metadata in filename.

    Parameters
    ----------
        config : Configs
            Configuration object.
        patches : np.ndarray
            MC-ILC patches to be saved.
        compsep_run : dict
            Dictionary with component separation parameters.
        nl_scale : int, optional
            Needlet scale index for the corresponding ILC run.

    Returns
    -------
        None
            It saves the MC-ILC patches to disk in the specified directory structure
    """
    path_ = os.path.join(compsep_run["path_out"], "patches")
    os.makedirs(path_, exist_ok=True)
    filename = os.path.join(path_, 
            f"patches_{compsep_run['field']}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}")
    if nl_scale is not None:
        filename += f"_nl{nl_scale}"
    np.save(filename, patches)

def save_spectra(
    config: Configs,
    cls_out: SimpleNamespace,
    compute_cls: Dict[str, Any],
    nsim: Optional[str] = None
) -> None:
    """
    Save the computed spectra to files based on the configuration and compute_cls dictionary.

    Parameters
    ----------
        config : Configs
            Configuration object containing parameters for spectra computation. See `_compute_spectra` for details.
        cls_out : SimpleNamespace
            Object containing computed spectra with attributes for each component.
        compute_cls : dict
            Dictionary containing parameters for spectra computation. See `_compute_spectra` for details.
        nsim : str, optional
            Simulation number to save spectra.

    Returns
    -------
        None
            It saves the computed spectra to disk in the specified directory structure.
    """

    path_spectra = get_path_spectra(config, compute_cls)

    post_filename = f"_{nsim}" if nsim is not None else ""

    pre_filename = "Dls" if config.return_Dell else "Cls"

    for component in compute_cls["components_for_cls"]:
        component_name = component.split('/')[0] if '/' in component else component
        os.makedirs(os.path.join(path_spectra, component), exist_ok=True)
        filename = os.path.join(
            path_spectra,
            f"{component}/{pre_filename}_{config.field_cls_out}_{component_name}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}{post_filename}.fits"
        )
        hp.write_cl(filename, getattr(cls_out, component_name), overwrite=True)

def get_path_spectra(config: Configs, compute_cls: Dict[str, Any]) -> str:
    """
    Get the path where spectra will be saved based on the compute_cls dictionary.

    Parameters
    ----------
        config : Configs
            Configuration object containing paths and parameters for spectra computation.
        compute_cls : dict
            Dictionary containing parameters for spectra computation, including mask type and fsky.

    Returns
    -------
        str
            Full path to the directory where spectra will be saved.

    """

    path_spectra = os.path.join(compute_cls["path"], 'spectra')
    mask_patterns = ['GAL*+fgres', 'GAL*+fgtemp', 'GAL*+fgtemp^3','GAL*0', 'GAL97', 'GAL99', 'fgres', 'fgtemp', 
            'fgtemp^3', 'config+fgres', 'config+fgtemp', 'config+fgtemp^3', 'config']

    if compute_cls["mask_type"] is None and config.mask_path is None:
        mask_name = 'fullsky'
    elif compute_cls["mask_type"] is None and config.mask_path is not None:
        mask_name = "fullpatch"
    elif any(fnmatch.fnmatch(compute_cls["mask_type"], pattern) for pattern in mask_patterns):
        if 'fgres' in compute_cls["mask_type"] or 'fgtemp' in compute_cls["mask_type"]:
            mask_name = compute_cls["mask_type"] + f"_fsky{compute_cls['fsky']}"
            if "smooth_tracer" in compute_cls:
                mask_name += f"_{compute_cls['smooth_tracer']}deg"
        elif compute_cls["mask_type"] == "config":
            mask_name = "fullpatch"
        else:
            mask_name = compute_cls["mask_type"]
    elif compute_cls["mask_type"] == "from_fits":
        mask_name = compute_cls.setdefault("mask_definition", "masks_from_fits")

    if compute_cls["apodize_mask"] is not None:
        mask_name += f"_apo{compute_cls['apodize_mask']}_{compute_cls['smooth_mask']}deg"

    return os.path.join(path_spectra, mask_name)
    
def _save_mask(mask: np.ndarray, 
               config: Configs, 
               compute_cls: Dict[str, Any], 
               nsim: Optional[str] = None) -> None:
    """
    Save the mask used for power spectra computation.

    Parameters
    ----------
        mask : np.ndarray
            The mask(s) to be saved.
        config : Configs
            Configuration object containing global parameters.
        compute_cls : dict
            Dictionary containing parameters for spectra computation.
        nsim : str, optional
            Simulation number to save the mask.

    Returns
    -------
        None
            It saves the mask to disk in the specified directory structure based on the configuration and compute_cls parameters.
    """
    
    path_mask = get_path_spectra(config, compute_cls)
    
    post_filename = f"_{nsim}" if nsim is not None else ""

    os.makedirs(path_mask, exist_ok=True)
    
    filename = f"mask_{config.field_cls_out}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}{post_filename}.fits"
    
    hp.write_map(os.path.join(path_mask, filename), mask, overwrite=True)

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]
                    

