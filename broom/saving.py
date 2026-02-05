import os
import re
import numpy as np
import healpy as hp
from typing import Any, Optional, Union, Dict, List
from .routines import _log, _format_nsim
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

    if compsep_run["method"] in ["cilc", "c_ilc", "mc_cilc","cpilc", "c_pilc","cprilc", "c_prilc"]:
        if 'mixed' in compsep_run["path_out"]:
            with open(os.path.join(compsep_run["path_out"], "constraints_info.txt"), "w") as f: 
                f.write("# Components and deprojection constraints used in the component separation run\n\n")
                
                components_header = "# Components per needlet band (if applicable):\n"
                f.write(components_header)
                if compsep_run["domain"] == "needlet":
                    for row in compsep_run["constraints"]["components"]:
                        f.write(" ".join(row) + "\n")
                else:
                    f.write(" ".join(compsep_run["constraints"]["components"]) + "\n")
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
        if "total" in attr_name:
            label_out = f"output_{attr_name}"
        elif attr_name == "m":
            label_out = "fgd_complexity"
        elif "cmb" in attr_name:
            if compsep_run["component_out"] == "cmb":
                label_out = f"output_{attr_name}"
            else:
                label_out = f"{attr_name}_residuals"
        elif "tsz" in attr_name:
            if compsep_run["component_out"] == "tsz":
                label_out = f"output_{attr_name}"
            else:
                label_out = f"{attr_name}_residuals"
        else:
            label_out = f"{attr_name}_residuals"
        

        path_c = os.path.join(compsep_run["path_out"], f"{label_out}")
        os.makedirs(path_c, exist_ok=True)

        if compsep_run["method"] in ["gilc","gpilc","gprilc"]:
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
    nsim: Optional[str] = None,
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

    gnilc_run = (re.search(r'(gilc_[^/]+)', compsep_run["gilc_path"])).group(1)
    if "needlet" in gnilc_run:
        folder_after = (compsep_run["gilc_path"]).split(gnilc_run + "/")[1].split("/")[0]
        gnilc_run += f"_{folder_after}"

    for attr_name, attr_values in vars(output_maps).items():
        if "total" in attr_name:
            label_out = "fgres_templates"
            if "split1" in attr_name:
                label_out += "_split1"
            elif "split2" in attr_name:
                label_out += "_split2"
        elif "fgds" in attr_name:
            label_out = "fgres_templates_ideal"
            if "split1" in attr_name:
                label_out += "_split1"
            elif "split2" in attr_name:
                label_out += "_split2"
        else:
            label_out = f"fgres_templates_{attr_name}"

        path_c = os.path.join(path_out, f"{label_out}", gnilc_run)
        os.makedirs(path_c, exist_ok=True)

        filename = (
            f"{config.field_out}_{label_out}_{config.fwhm_out}acm_"
            f"ns{config.nside}_lmax{config.lmax}"
        )
        if compsep_run["nsim_weights"] is not None and nsim != compsep_run["nsim_weights"]:
            filename += f"_w{compsep_run['nsim_weights']}"
        if nsim is not None:
            filename += f"_{nsim}"
        filename += ".fits"

        hp.write_map(os.path.join(path_c, filename), attr_values, overwrite=True)

def _save_combination(
    config: Configs,
    output_maps: SimpleNamespace,
    compsep_run: Dict[str, Any],
    nsim: Optional[str] = None,
) -> None:
    """
    Save output maps from combination of input data with component separation weights.

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
            It saves the output maps from combination to disk in the specified directory structure
            based on the component separation method and configuration.
    
    """
    path_out = os.path.join(config.path_outputs, compsep_run["compsep_path"])

    for attr_name, attr_values in vars(output_maps).items():
        label_out = f"propagated_{attr_name}"

        if "extra_info" in compsep_run:
            path_c = os.path.join(path_out, f"{label_out}_{compsep_run['extra_info']}")
        else:
            path_c = os.path.join(path_out, f"{label_out}")
        
        if compsep_run["method"] == "gilc":
            if nsim is not None:
                path_c = os.path.join(path_c, f"{nsim}")

        os.makedirs(path_c, exist_ok=True)

        if compsep_run["method"] == "ilc":
            filename = (
                f"{config.field_out}_{label_out}_{config.fwhm_out}acm_"
                f"ns{config.nside}_lmax{config.lmax}"
            )
            if compsep_run["nsim_weights"] is not None and nsim != compsep_run["nsim_weights"]:
                filename += f"_w{compsep_run['nsim_weights']}"
            if nsim is not None:
                filename += f"_{nsim}"
            filename += ".fits"

            hp.write_map(os.path.join(path_c, filename), attr_values, overwrite=True)
        else:
            for f, freq in enumerate(compsep_run["channels_out"]):
                tag = config.instrument.channels_tags[freq]
                filename = (
                    f"{config.field_out}_{label_out}_{tag}_{config.fwhm_out}acm_"
                    f"ns{config.nside}_lmax{config.lmax}"
                )
                if compsep_run["nsim_weights"] is not None and nsim != compsep_run["nsim_weights"]:
                    filename += f"_w{compsep_run['nsim_weights']}"
                if nsim is not None:
                    filename += f"_{nsim}"
                filename += ".fits"

                hp.write_map(os.path.join(path_c, filename), attr_values[f], overwrite=True)


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
    
    if compsep_run["method"] in ["mc_ilc", "mc_cilc", "c_ilc", "c_pilc", "c_prilc"]:
        complete_path = f'{compsep_run["method"]}_{compsep_run["domain"]}_bias{compsep_run["ilc_bias"]}_nls{"-".join(map(str, compsep_run["special_nls"]))}' 
    elif compsep_run["method"] == "mcilc":
        complete_path = f'{compsep_run["method"]}_{compsep_run["domain"]}'
    elif compsep_run["method"] in ["gilc", "gpilc", "gprilc"]:
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

    if compsep_run["method"] in ["gilc", "gpilc", "gprilc", "fgd_diagnostic", "fgd_P_diagnostic"]:
        if isinstance(compsep_run["nuisance"], str):
            complete_path += f"_nuis_{compsep_run['nuisance']}"
        elif isinstance(compsep_run["nuisance"], list):
            if len(compsep_run["nuisance"]) == 1:
                complete_path += f"_nuis_{compsep_run['nuisance'][0]}"
            else:
                nuis_text = "_nuis_"
                for idx, nuis in enumerate(compsep_run["nuisance"]):
                    if idx == len(compsep_run["nuisance"]) - 1:
                        nuis_text += f"{nuis}"
                    else:
                        nuis_text += f"{nuis}+"
                complete_path += nuis_text

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

    if compsep_run["method"] in ["cilc", "c_ilc", "mc_cilc","cpilc", "c_pilc","cprilc", "c_prilc"]:
        if compsep_run["domain"] == "pixel":
            mom_text = "".join(compsep_run["constraints"]["components"])
        elif compsep_run["domain"] == "needlet":
            if all(list(set(row)) == list(set(compsep_run["constraints"]["components"][0])) for row in compsep_run["constraints"]["components"]):
                mom_text = "".join(compsep_run["constraints"]["components"][0])
            else:
                mom_text = ""
                for idx, row in enumerate(compsep_run["constraints"]["components"]):
                    if idx == 0:
                        mom_text += "".join(row)
                    else:
                        if list(set(row)) != list(set(compsep_run["constraints"]["components"][idx-1])):
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

#        complete_path = os.path.join(complete_path, mom_text)

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
        if compsep_run["b_squared"] or compsep_run["method"] in ["pilc", "cpilc", "c_pilc", "gpilc", "prilc", "cprilc", "c_prilc", "gprilc", "fgd_P_diagnostic"]:
            text_ += "_nlsquared"
        complete_path = os.path.join(complete_path, text_)

    if compsep_run["method"] in ["mcilc","mc_ilc","mc_cilc"]:
        text_ = compsep_run["mc_type"]
        for freq_tracer in compsep_run["channels_tracers"]:
            text_ += f"_{config.instrument.channels_tags[freq_tracer]}"
        text_ += f"_{compsep_run['n_patches']}patches"
        complete_path = os.path.join(complete_path, text_)

    if compsep_run["method"] not in ["fgd_diagnostic","fgd_P_diagnostic", "gilc", "gpilc", "gprilc"]:
        comp_out = compsep_run["component_out"]
        if compsep_run["component_out"] in ['0d', '1bd', '1Td', '2bd', '2Td', '2bdTd', '2Tdbd']:
            if "beta_d_out" not in compsep_run or "T_d_out" not in compsep_run or "nu_ref_d_out" not in compsep_run:
                raise ValueError(f"compsep_run must contain 'beta_d_out', 'T_d_out' and 'nu_ref_d_out' for component_out '{comp_out}'")
            comp_out += f"_{compsep_run['nu_ref_d_out']}GHz_bd{compsep_run['beta_d_out']}_Td{compsep_run['T_d_out']}"
        elif compsep_run["component_out"] in ['0s', '1bs', '2bs']:
            if "beta_s_out" not in compsep_run or "nu_ref_s_out" not in compsep_run:
                raise ValueError(f"compsep_run must contain 'beta_s_out' and 'nu_ref_s_out' for component_out '{comp_out}'")
            comp_out += f"_{compsep_run['nu_ref_s_out']}GHz_bs{compsep_run['beta_s_out']}"
        comp_out += "_reconstruction"

        if compsep_run["from_splits"]:
            comp_out += "_fromsplits"
        if compsep_run["method"] in ["cilc", "c_ilc", "mc_cilc","cpilc", "c_pilc","cprilc", "c_prilc"]:
            comp_out += f"_{mom_text}"
        
        complete_path = os.path.join(complete_path, comp_out)

    path_out = os.path.join(config.path_outputs, complete_path)

    return path_out

def _get_full_path_nuiscov(config: Configs, compsep_run: Dict[str, Any]) -> str:
    """
    Constructs the full output path for saving nuisance covariance matrices based on configuration and run options.
    Parameters
    -----------
        config: Configs
            Configuration object.
        compsep_run: dict
            Dictionary containing method and domain setup.

    Returns
    --------
        str
            Full path where output nuisance covariance matrices should be saved.
    """
    
    complete_path = f'nuisance_covariances_{compsep_run["domain"]}_bias{compsep_run["ilc_bias"]}'

    if (config.leakage_correction is not None) and ("QU" in config.field_in) and (config.mask_observations is not None):
        leak_def = (config.leakage_correction).split("_")[0] + (config.leakage_correction).split("_")[1] 
        if "_recycling" in config.leakage_correction:
            if "_iterations" in config.leakage_correction:
                iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
                leak_def += f'_iters{iterations}'
        complete_path += f"_{leak_def}"

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
        if compsep_run["b_squared"]:
            text_ += "_nlsquared"
        complete_path = os.path.join(complete_path, text_)

    path_out = os.path.join(config.path_outputs, complete_path)

    return path_out


def get_gilc_maps(
    config: Configs,
    gilc_config: Dict[str, Any],
    nsim: Optional[str] = None
) -> SimpleNamespace:
    """
    Load GNILC component separation results (total signal, noise residuals, and foreground residuals) of all frequency channels
    provided in the instrument object and for a given simulation run and field. 

    Parameters
    -----------
        config : Configs
            Configuration object containing instrument and output specifications.
        gilc_config : Dict[str, Any]
            Dictionary containing GNILC configuration parameters, including:
                path_gilc : str
                    Root path to the GNILC output directory. The full path will be given by '{config.path_outputs}/{path_gilc}'.
                gilc_components : List[str]
                    List of GNILC components to load. Possible elements are "output_total", "noise_residuals", "fgds_residuals".
                field_in : Optional[str], default=None
                    Type of field to load ("T", "QU", "EB", "TQU", "TEB", etc.). If None, default is config.field_out.
        nsim : Optional[Union[str, int]], default=None
            Simulation identifier, if any (used to select specific simulation output files).

    Returns
    --------
        gnilc_maps : SimpleNamespace
            A container of numpy arrays with attributes corresponding to the requested GNILC components:
    """
    path_gilc = gilc_config["gilc_path"]
    gilc_components = gilc_config["gilc_components"]
    nside = gilc_config["nside"]
    fwhm_out = gilc_config["fwhm_out"]
    lmax = gilc_config["lmax"]
    field_in = gilc_config["field_in"]

    if not os.path.exists(os.path.join(config.path_outputs, path_gilc)):
        raise ValueError(f"Path {os.path.join(config.path_outputs, path_gilc)} does not exist.")
    if field_in is None:
        field_in = config.field_out
    
    gilc_maps = SimpleNamespace()

    if field_in in ["TQU", "TEB"]:
        if config.field_out == "T":
            gilc_fields = 0
        elif config.field_out in ["QU", "QU_E", "QU_B", "E", "B"]:
            gilc_fields = (1,2)
        elif config.field_out in ["TQU","TEB"]:
            gilc_fields = (0,1,2)
    elif field_in in ["QU","EB"]:
        gilc_fields = (0,1)
    elif field_in in ["T","E","B"]:
        gilc_fields = 0

    for component in gilc_components:
        if component.split("_")[-1] == "residuals":
            attr_out = component.split("_residuals")[0]
        elif "output" in component:
            attr_out = component.split("output_")[1]

        filepath = os.path.join(config.path_outputs, path_gilc, component)
        if nsim is not None:
            filepath = os.path.join(filepath, nsim)
        if not os.path.exists(filepath):
            raise ValueError(f"Path {filepath} does not contain the expected GNILC {component} maps.")

        setattr(gilc_maps, attr_out, [])

        for f, freq in enumerate(config.instrument.frequency):
            tag = config.instrument.channels_tags[f]
            filename = f"{field_in}_{component}_{tag}_{fwhm_out}acm_ns{nside}_lmax{lmax}"
            if nsim is not None:
                filename += f"_{nsim}"
            filename += ".fits"
            getattr(gilc_maps, attr_out).append(hp.read_map(os.path.join(filepath, filename), field=gilc_fields))
        setattr(gilc_maps, attr_out, np.array(getattr(gilc_maps, attr_out)))

    return gilc_maps

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

def update_and_save_nuiscov_serial(
    config: Configs,
    cov_n: np.ndarray,
    compsep_run: Dict,
    nside_: int,
    nl_scale: Optional[Union[int, None]] = None
) -> None:
    """
    Load, update, and save nuisance covariance matrices to disk with appropriate metadata in filename.

    Parameters
    ----------
        config : Configs
            Configuration object.
        cov_n : np.ndarray
            Nuisance covariance matrix to be added to the average covariance.
        compsep_run : dict
            Dictionary with parameters for nuisance covariance computation.
        nside_ : int
            HEALPix NSIDE resolution of the nuisance covariance.
        nl_scale : int, optional
            Needlet scale index associated with the nuisance covariance.
    
    Returns
    -------
        None
            It loads, updates, and saves the nuisance covariance matrices to disk in the specified directory structure
    """
    path_cov = compsep_run["path_out"]
    os.makedirs(path_cov, exist_ok=True)

    if isinstance(compsep_run["nuisance"], str) or len(compsep_run["nuisance"]) == 1:
        filename = compsep_run["nuisance"] if isinstance(compsep_run["nuisance"], str) else compsep_run["nuisance"][0]
    else:
        default_nuis = [
            "cmb", "noise"
        ]
        nuis_attrs = [attr for attr in default_nuis if attr in compsep_run["nuisance"]]
        if any(x not in ["cmb", "noise"] for x in compsep_run["nuisance"]):
            nuis_fgds = [x for x in compsep_run["nuisance"] if x not in ["cmb", "noise"]]

        prefix_models = ["d", "s", "a", "co", "f", "cib", "tsz", "ksz", "rg"]
        for p in prefix_models:
            for model in nuis_fgds:
                if model.startswith(p):
                    nuis_attrs.append(model)

        filename = ""
        for idx, nuis in enumerate(nuis_attrs):
            if idx == len(nuis_attrs) - 1:
                filename += f"{nuis}"
            else:
                filename += f"{nuis}+"
        
    filename = os.path.join(path_cov, f"{filename}_covariance")
    if compsep_run["type"] == "Pr_scalar":
        filename += f"_+"
    filename += f"_{compsep_run['field']}_{config.fwhm_out}acm_ns{nside_}_lmax{config.lmax}"
    if nl_scale is not None:
        filename += f"_nl{nl_scale}"

    if compsep_run["nsim"] is None:
        np.save(filename, cov_n)
    else:
#        if int(compsep_run["nsim"]) == 0:
#            filename += f"_1sims"
#            np.save(filename, cov_n)
#        elif int(compsep_run["nsim"]) > 0:
#            filename_prev = filename + f"_{int(compsep_run['nsim'])}sims.npy"
#            filename += f"_{int(compsep_run['nsim']) + 1}sims"
#            if not os.path.exists(filename_prev):
#                raise FileNotFoundError(f"Nuisance covariance file {filename_prev} not found for updating.")
#            np.save(filename, (np.load(filename_prev) * int(compsep_run["nsim"]) + cov_n) / (int(compsep_run["nsim"]) + 1))
        files = [f for f in os.listdir(path_cov) if f.startswith(os.path.basename(filename)) and f.endswith(".npy")]

        def get_num_sims(f):
            m = re.search(r"_(\d+)sims\.npy$", f)
            return int(m.group(1)) if m else None

        max_sims = max([get_num_sims(f) for f in files], default=None)
        if max_sims is None:
            filename += f"_1sims"
            np.save(filename, cov_n)
        else:
            filename_prev = filename + f"_{int(max_sims)}sims.npy"
            filename += f"_{int(max_sims + 1)}sims"
            np.save(filename, (np.load(filename_prev) * int(max_sims) + cov_n) / (int(max_sims) + 1))
            os.remove(filename_prev)

    return None


def load_nuiscov(
    config: Configs,
    path_cov: str,
    compsep_run: Dict,
    nside_: int,
    nuisance: Union[str, List[str]],
    nl_scale: Optional[Union[int, None]] = None,
    ) -> None:
    """
    Load nuisance covariance matrices from disk.

    Parameters
    ----------
        config : Configs
            Configuration object.
        path_cov : str
            Path to the nuisance covariance matrix to be loaded.
        compsep_run : dict
            Dictionary with parameters for nuisance covariance computation.
        nside_ : int
            HEALPix NSIDE resolution of the nuisance covariance.
        nuisance : str or List[str]
            Nuisance component(s) to be included in the covariance. Can be a single string or a list of strings.
        nl_scale : int, optional
            Needlet scale index associated with the nuisance covariance.
    
    Returns
    -------
        None
            It loads, updates, and saves the nuisance covariance matrices to disk in the specified directory structure
    """
    if isinstance(nuisance, str) or len(nuisance) == 1:
        filename = nuisance if isinstance(nuisance, str) else nuisance[0]
    else:
        default_nuis = [
            "cmb", "noise"
        ]
        nuis_attrs = [attr for attr in default_nuis if attr in nuisance]
        if any(x not in ["cmb", "noise"] for x in nuisance):
            nuis_fgds = [x for x in nuisance if x not in ["cmb", "noise"]]

        prefix_models = ["d", "s", "a", "co", "f", "cib", "tsz", "ksz", "rg"]
        for p in prefix_models:
            for model in nuis_fgds:
                if model.startswith(p):
                    nuis_attrs.append(model)

        filename = ""
        for idx, nuis in enumerate(nuis_attrs):
            if idx == len(nuis_attrs) - 1:
                filename += f"{nuis}"
            else:
                filename += f"{nuis}+"

    filename = os.path.join(path_cov, f"{filename}_covariance")
    if compsep_run["method"] in ["gpilc", "prilc", "cprilc", "c_prilc", "gprilc", "fgd_P_diagnostic"]:
        filename += "_+"
    filename += f"_{compsep_run['field']}_{config.fwhm_out}acm_ns{nside_}_lmax{config.lmax}"
    if nl_scale is not None:
        filename += f"_nl{nl_scale}"

    files = [f for f in os.listdir(path_cov) if f.startswith(os.path.basename(filename)) and f.endswith(".npy")]

    def get_num_sims(f):
        m = re.search(r"_(\d+)sims\.npy$", f)
        return int(m.group(1)) if m else None

    max_sims = max([get_num_sims(f) for f in files], default=None)
    if max_sims is None:
        _log(f"No nuisance covariance found from nuisance simulations. Looking for file: {filename}.npy", config.verbose)
        filename += ".npy"
    else:
        filename += f"_{int(max_sims)}sims.npy"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Nuisance covariance file {filename} not found.")
    return np.load(filename)

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
        if isinstance(component, str):
            if "gilc_" in compute_cls["path"] or "gpilc_" in compute_cls["path"] or "gprilc_" in compute_cls["path"]:
                component_name = '_'.join(component.split('_')[:-1])
                if nsim is not None:
                    os.makedirs(os.path.join(path_spectra, component_name, nsim), exist_ok=True)
                    filename = os.path.join(
                        path_spectra,
                        f"{component_name}/{nsim}/{pre_filename}_{config.field_cls_out}_{component}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}{post_filename}.fits"
                    )
                else:
                    os.makedirs(os.path.join(path_spectra, component_name), exist_ok=True)
                    filename = os.path.join(
                        path_spectra,
                        f"{component_name}/{pre_filename}_{config.field_cls_out}_{component}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}{post_filename}.fits"
                    )
                hp.write_cl(filename, getattr(cls_out, component), overwrite=True)
            else:
                component_name = component.split('/')[0] if '/' in component else component
                os.makedirs(os.path.join(path_spectra, component), exist_ok=True)
                filename = os.path.join(
                    path_spectra,
                    f"{component}/{pre_filename}_{config.field_cls_out}_{component_name}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}{post_filename}.fits"
                )
                hp.write_cl(filename, getattr(cls_out, component_name), overwrite=True)
        elif isinstance(component, list):
            if "gilc_" in compute_cls["path"] or "gpilc_" in compute_cls["path"] or "gprilc_" in compute_cls["path"]:
                for idx, component_ in enumerate(component):
                    if idx==0:
                        component_name_ = '_'.join(component_.split('_')[:-1])
                        component_name = component_
                    elif idx == 1:
                        component_name_ = f"_x_{'_'.join(component_.split('_')[:-1])}"
                        component_name += f"_x_{component_}"
                if nsim is not None:
                    os.makedirs(os.path.join(path_spectra, component_name_, nsim), exist_ok=True)
                    filename = os.path.join(
                        path_spectra,
                        f"{component_name_}/{nsim}/{pre_filename}_{config.field_cls_out}_{component_name}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}{post_filename}.fits"
                    )
                else:
                    os.makedirs(os.path.join(path_spectra, component_name_), exist_ok=True)
                    filename = os.path.join(
                        path_spectra,
                        f"{component_name_}/{pre_filename}_{config.field_cls_out}_{component_name}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}{post_filename}.fits"
                    )
                hp.write_cl(filename, getattr(cls_out, component_name), overwrite=True)
            else:
                component_name = component[0].split('/')[0] if '/' in component[0] else component[0]
                component_name += f"_x_{component[1].split('/')[0] if '/' in component[1] else component[1]}"
                second_path = component_name
                if '/' in component[0] and '/' in component[1]:
                    second_path += f"/{component[0].split('/')[1]}_x_{component[1].split('/')[1]}"
                elif '/' in component[0]:
                    second_path += f"/{component[0].split('/')[1]}"
                elif '/' in component[1]:
                    second_path += f"/{component[1].split('/')[1]}"
                os.makedirs(os.path.join(path_spectra, second_path), exist_ok=True)
                filename = os.path.join(
                    path_spectra,
                    f"{second_path}/{pre_filename}_{config.field_cls_out}_{component_name}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}{post_filename}.fits"
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
            'fgtemp^3']

    if compute_cls["mask_type"] is None and config.mask_observations is None and config.mask_covariance is None:
        mask_name = 'fullsky'
    if config.mask_observations is not None and config.mask_covariance is None:
        mask_name = "obspatch"
    if config.mask_covariance is not None:
        mask_name = f"{os.path.basename(config.mask_covariance).split('.fits')[0]}"
    
    if compute_cls["mask_type"] is not None:
        if any(fnmatch.fnmatch(compute_cls["mask_type"], pattern) for pattern in mask_patterns):
            if 'fgres' in compute_cls["mask_type"] or 'fgtemp' in compute_cls["mask_type"]:
                mask_name_ = compute_cls["mask_type"] + f"_fsky{compute_cls['fsky']}"
                if "smooth_tracer" in compute_cls:
                    mask_name_ += f"_{compute_cls['smooth_tracer']}deg"
            #elif compute_cls["mask_type"] == "config":
            #    mask_name = "fullpatch"
            else:
                mask_name_ = compute_cls["mask_type"]
        elif compute_cls["mask_type"] == "from_fits":
            #mask_name = compute_cls.setdefault("mask_definition", "masks_from_fits")
            mask_name_ = os.path.basename(compute_cls["mask_path"]).split('.fits')[0]
        if config.mask_observations is None and config.mask_covariance is None:
            mask_name = mask_name_
        else:
            mask_name += f"_u_{mask_name_}"


#    if compute_cls["mask_type"] is None and config.mask_observations is None and config.mask_covariance is None:
    if mask_name != 'fullsky':
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
                    

