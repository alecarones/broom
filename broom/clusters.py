import numpy as np 
import healpy as hp
import os
from .routines import _slice_data, _map2alm_kwargs, _log
from .needlets import _get_needlet_windows_,  _needlet_filtering
from .configurations import Configs
from types import SimpleNamespace
from .simulations import _get_data_foregrounds_, _get_data_simulations_
from typing import Optional, Union, List, Dict, Any, Tuple
import sys


def get_and_save_real_tracers_B(
    config: Configs,
    foregrounds: Optional[np.ndarray] = None,
    systematics: Optional[np.ndarray] = None,
    **kwargs: Any
) -> None:
    """
    Generate and save realistic tracer for MC-ILC given the instrumental and parameters configuration.

    Parameters
    ----------
        config : Configs
            Configuration object containing the instrumental and parameters configuration. 
            It should contain the following attributes:
            - experiment : str
                Name of the experiment
            - lmax : int 
                Maximum multipole for the simulations and analysis.      
            - lmin : int
                Minimum multipole for the simulations and analysis.
            - nside : int
                HEALPix resolution parameter.   
            - data_type : str
                Type of data to be used in the simulations, either "maps" or "alms". 
                It also identifies the type of data associated eventually to provided foregrounds and systematics.
            - fwhm_out : float
                Full width at half maximum of the output beam in arcminutes used to generate the tracer.
            - foreground_models : list
                List of foreground models to be used in the simulations. Used if foregrounds is None.
            - real_mc_tracers : dict
                List of dictionaries containing the information about the MC-ILC tracers to be generated.
                It should contain the following keys:
                   - channels_tracers: list
                        List of channels indexes to be used for the tracers generation. 
                   - path_tracers : str
                        Path where the tracers will be saved.
            - bandpass_integrate : bool
                Boolean indicating whether to integrate over the bandpass of the instrument in the simulations.
            - instrument : dict
                Instrument configuration object containing the following attributes:
                   - channels_tags: list
                        List of tags for the instrument channels.
            - mask_observations : str
                Path to the mask which sets to zero unobserved pixels. If None, no mask is applied.
            - coordinates : str
                Coordinate system used for the simulations, either "G" for Galactic, "E" for Ecliptic, or "C" for Celestial.
            - pixel_window_in : bool
                Boolean indicating whether to apply the pixel window function to the input data.
            - pixel_window_out : str 
                Boolean indicating whether to apply the pixel window function to the output tracer.
            - verbose : bool
                Boolean indicating whether to print information about the process.
            
        foregrounds : np.ndarray, optional
            Foregrounds data to be used in the simulations for tracer generation. If None, the foregrounds will be generated using the configuration parameters.
            Shape depends on `config.data_type`:
            - If "maps": (n_channels, 3, n_pixels) for (T, Q, U)
            - If "alms": (n_channels, 3, n_alms) for (T, E, B) 
        
            
        systematics : np.ndarray, optional
            Systematics data to be added to the simulations for tracer generation. If None, no systematics are added.
            If provided, it should be a 3D array with shape: 
        (n_Shape depends on `config.data_type`:
            - If "maps": (n_channels, 3, n_pixels) for (T, Q, U)
            - If "alms": (n_channels, 3, n_alms) for (T, E, B) 

        kwargs : dict, optional
            Additional keyword arguments to be passed to healpy 'map2alm' function.

    Returns
    -------
        None
            Saves the generated tracers to the configured output path.
    """
    from broom import component_separation
    if "tracers_inputs_path" not in config.real_mc_tracers[0]:
        config.real_mc_tracers[0]["tracers_inputs_path"] = f"inputs_mc_tracers/{config.experiment}"

    config_mc = get_mc_config(config, config.real_mc_tracers[0]["tracers_inputs_path"])
    kwargs = _map2alm_kwargs(**kwargs)

    _log("Generating input simulations for MC-ILC tracers", verbose=config_mc.verbose)

    mc_data = get_mc_data(config_mc, foregrounds=foregrounds, systematics=systematics, **kwargs)

#    if systematics is not None:
#        print("Adding systematic effect to the data")
#        systematics = _adapt_systamatics(
#            systematics, mc_data.total.shape, config.lmax,
#            data_type=config_mc.data_type, **kwargs
#        )
#        mc_data.total = mc_data.total + systematics

    config_mc.compsep = get_tracers_compsep(config.real_mc_tracers[0]["channels_tracers"], config.lmax)

    _log(f"Generating the MC-ILC tracers for {config.experiment} experiment and {''.join(config.foreground_models)} foreground model", verbose=config_mc.verbose)

    #mc_data.nuisance = mc_data.noise + mc_data.cmb

    tracers = component_separation(config_mc, mc_data)
    
    tracers = _combine_B_tracers(np.array(tracers.total))

    _log(f"Saving the tracers in {config.real_mc_tracers[0]['path_tracers']} directory", verbose=config_mc.verbose)

    _save_real_tracers_B(
        tracers,
        config.real_mc_tracers[0]["path_tracers"],
        np.array(config_mc.instrument.channels_tags)[config.real_mc_tracers[0]["channels_tracers"]],
        config_mc.fwhm_out,
        config_mc.lmax
    )

def _save_real_tracers_B(tracers, path_tracers, tags, fwhm_out, lmax):
    """
    Save the generated realistic B-mode tracers in the specified path.

    Parameters
    ----------
        tracers : np.ndarray
            The generated B-mode tracers to be saved. It should be a 2D with shape (n_channels, n_alms).
        path_tracers : str
            The path where the tracers will be saved. 
        tags : List[str]
            List of tags for the tracers, corresponding to the frequency channels of the tracers.
        fwhm_out : float
            Full width at half maximum of the output beam in arcminutes associated to the tracers.
        lmax : int
            Maximum multipole for the tracers.
    
    Returns
    -------
        None
            Saves the tracers in the specified path with the format "B_tracer_{tag}_{fwhm_out}acm_ns{nside}_lmax{lmax}.fits"

    """

    if not os.path.exists(path_tracers):
        os.makedirs(path_tracers)

    if not path_tracers.endswith('/'):
        path_tracers = path_tracers + '/'

    for i, tracer in enumerate(tracers):
        hp.write_map(path_tracers + f"B_tracer_{tags[i]}_{fwhm_out}acm_ns{hp.npix2nside(tracer.shape[0])}_lmax{lmax}.fits", tracer, overwrite=True)
    
def initialize_scalar_tracers(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: Dict[str, Any],
    field: str = "B",
    **kwargs: Any
) -> np.ndarray:
    """
    Return the scalar tracers for the MC-ILC component separation run.

    Parameters
    ----------
        config : Configs
            Configuration object containing the instrumental and parameters configuration. It should contain the following attributes:
            - `nside`: HEALPix resolution parameter.
            - `lmax`: Maximum multipole for the tracers.
        input_alms : np.ndarray
            Input foreground alms from which the tracers will be derived if `compsep_run["mc_type"]` is "cea_ideal" or "rp_ideal".
            It should have shape (n_channels, n_alms).
        compsep_run : dict
            Dictionary containing the information about the component separation run. It should contain the following keys:
            - `mc_type`: Type of MC-ILC tracer and partition, either "cea_ideal", "rp_ideal", "cea_real", or "rp_real".
            - `domain`: Domain where component separation is performed, either "needlet" or "pixel".
            - `channels_tracers`: List of channels indexes to be used for the tracers generation or loading.
            - `path_tracers`: Path where the tracers will be loaded from if `compsep_run["mc_type"]` is "cea_real" or "rp_real".
        field : str, optional
            Field of the tracers to be loaded or derived, only "B" is supported. Default is "B".
        kwargs : dict, optional
            Additional keyword arguments to be passed to healpy 'map2alm' function.
    
    Returns
    -------
        tracers : np.ndarray
            The scalar tracers for the MC-ILC component separation run. 
            If compsep_run["domain"] is "needlet", it will be a 2D array with alms of the tracers.
            If compsep_run["domain"] is "pixel", it will be a 2D array with maps of the tracers.
    """

    if compsep_run["mc_type"] in ["cea_ideal", "rp_ideal"]:
        if compsep_run["domain"] == "needlet":
            tracers = np.array([
                input_alms[compsep_run["channels_tracers"][0], :],
                input_alms[compsep_run["channels_tracers"][1], :]
            ])
        elif compsep_run["domain"] == "pixel":
            tracers = np.array([
                hp.alm2map(input_alms[compsep_run["channels_tracers"][0], :], config.nside, lmax=config.lmax, pol=False),
                hp.alm2map(input_alms[compsep_run["channels_tracers"][1], :], config.nside, lmax=config.lmax, pol=False)
            ])
    if compsep_run["mc_type"] in ["cea_real", "rp_real"]:
        tracers_tags = [config.instrument.channels_tags[freq] for freq in compsep_run["channels_tracers"]]
        tracer_paths = get_tracers_paths_for_ratio(config, compsep_run["path_tracers"], tracers_tags, field=field)
        if compsep_run["domain"] == "needlet":
            tracers = load_scalar_tracers_for_ratio(tracer_paths, config.nside, config.lmax, return_alms=True, **kwargs)
        elif compsep_run["domain"] == "pixel":
            tracers = load_scalar_tracers_for_ratio(tracer_paths, config.nside, config.lmax, return_alms=False, **kwargs)
    return tracers

def get_tracers_paths_for_ratio(
    config: Configs,
    path_tracers: str,
    tracers_tags: List[str],
    field: str = "B"
) -> List[str]:
    """
    Get the paths to the scalar tracers for the MC-ILC component separation run.

    Parameters
    ----------
        config : Configs
            Configuration object containing the instrumental and parameters configuration. It should contain the following attributes:
            - `nside`: HEALPix resolution parameter.
            - `lmax`: Maximum multipole for the tracers.
            - `fwhm_out`: Full width at half maximum of the output beam in arcminutes associated to the tracers.
        path_tracers : str
            Path where the tracers are stored. It should be a directory path.
        tracers_tags : List[str]
            List of tags for the tracers, corresponding to the frequency channels of the tracers.
        field : str, optional
            Field of the tracers to be loaded, only "B" is supported. Default is "B".
    
    Returns
    -------
        tracers_paths : List[str]
            List of paths to the scalar tracers for the MC-ILC component separation run.
    """

    tracers_paths = [
        f"{path_tracers}{field}_tracer_{tag}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}.fits"
        for tag in tracers_tags
    ]

    missing_tracers = [tracers_tags[n] for n, path in enumerate(tracers_paths) if not os.path.exists(path)]

    if missing_tracers:
        raise ValueError(f"Missing tracer files: {missing_tracers}. Please check the paths or run the tracer generation routine.")

    return tracers_paths

def load_scalar_tracers_for_ratio(
    tracers_paths: List[str],
    nside: int,
    lmax: int,
    field: str = "B",
    return_alms: bool = True,
    **kwargs: Any
) -> np.ndarray:
    """
    Load the scalar tracers from the specified paths.

    Parameters
    ----------
        tracers_paths : List[str]
            List of paths to the scalar tracers to be loaded.
        nside : int
            HEALPix resolution parameter for the tracers.
        lmax : int
            Maximum multipole for the tracers.
        field : str, optional
            Field of the tracers to be loaded, only "B" is supported. Default is "B".
        return_alms : bool, optional
            Whether to return the tracers as alms or maps. If True, returns alms; if False, returns maps. Default is True.
        kwargs : dict, optional
            Additional keyword arguments to be passed to healpy 'map2alm' function.
    
    Returns
    -------
        tracers : np.ndarray
            The loaded scalar tracers. If `return_alms` is True, it will be a 2D array with alms of the tracers.
            If `return_alms` is False, it will be a 2D array with maps of the tracers.
    """

    tracers = []
    for tracer_path in tracers_paths:
        tracer = hp.read_map(tracer_path, field=0)
        if return_alms:
            tracer = hp.map2alm(tracer, lmax=lmax, pol=False, **kwargs)
        else:
            if hp.get_nside(tracer) != nside:
                alm_ = hp.map2alm(tracer, lmax=lmax, pol=False, **kwargs)
                tracer = hp.alm2map(alm_, nside, lmax=lmax, pol=False)
        tracers.append(tracer)

    return np.array(tracers)     

def get_scalar_tracer_nl(tracers, nside_, lmax_, b_ell):
    """
    Get the scalar MC-ILC ratio tracer in the needlet domain.

    Parameters
    ----------
        tracers : np.ndarray
            The scalar tracers to be filtered. It can be either a 1D array (single tracer) or a 2D array (multiple tracers).
            If 2D, the ratio of the first tracer to the second tracer will be returned.
        nside_ : int
            HEALPix resolution parameter for the tracers.
        lmax_ : int
            Maximum multipole for the tracers.
        b_ell : np.ndarray
            Needlet filter bandpass function to be applied to the tracers. 
    
    Returns
    -------
        np.ndarray
            The filtered scalar tracer in the needlet domain. 
    """

    if tracers.ndim == 1:
        alm_tracer_nl = _needlet_filtering(tracers, b_ell, lmax_)
        return hp.alm2map(alm_tracer_nl, nside_, lmax=lmax_, pol=False)
    elif tracers.ndim == 2:
        tracers_nl = []
        for tracer in tracers:
            alm_tracer_nl = _needlet_filtering(tracer, b_ell, lmax_)
            tracers_nl.append(hp.alm2map(alm_tracer_nl, nside_, lmax=lmax_, pol=False))
        tracers_nl = np.array(tracers_nl)
        return tracers_nl[0] / tracers_nl[1]

def get_scalar_tracer(
    tracers: np.ndarray
) -> np.ndarray:
    """
    Get the scalar MC-ILC ratio tracer in the pixel domain.

    Parameters
    ----------
        tracers : np.ndarray
            The scalar tracers. It can be either a 1D array (single tracer) or a 2D array (multiple tracers).
            If 2D, the ratio of the first tracer to the second tracer will be returned.
    
    Returns
    -------
        np.ndarray
            The scalar tracer in the pixel domain. If `tracers` is a 1D array, it returns the tracer as is.
    """

    if tracers.ndim == 1:
        return tracers
    elif tracers.ndim == 2:
        return tracers[0] / tracers[1]

def _cea_partition(
    map_: np.ndarray,
    n_patches: int,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Partition the input map into n_patches using the CEA method.
    The CEA method splits the map into n_patches patches with equal number of pixels based on the values of the map.

    Parameters
    ----------
        map_ : np.ndarray
            The input map to be partitioned. It should be a 1D array of pixel values.
        n_patches : int
            The number of patches to be created from the input map.
        mask : np.ndarray, optional
            A mask to be applied to the input map. If provided, only the pixels where the mask is non-zero will be considered for partitioning.
            If None, the entire map will be used for partitioning.
    
    Returns
    -------
        np.ndarray
            A 1D array of the same shape as the input map, where each pixel is assigned a patch number from 0 to n_patches-1.
    """
    if mask is None:
        mask = np.ones_like(map_)

    split = np.array_split(np.sort(map_[mask > 0.]),n_patches)
    patches = np.zeros(12 * (hp.get_nside(map_))**2)
    
    for n in range(n_patches):
        if n==0:
            patches[(map_ <= max(split[n])) & (mask > 0.)] = float(n)
        elif n==(n_patches-1):
            patches[(map_ >= min(split[n])) & (mask > 0.)] = float(n)
        else:
            patches[(min(split[n]) <= map_) & (map_ <= max(split[n])) & (mask > 0.)] = float(n)
            
    return patches

def _rp_partition(map_: np.ndarray,
    n_patches: int,
    mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Partition the input map into n_patches using the RP method.
    The RP method splits the map into n_patches patches with random number of pixels based on the values of the map.

    Parameters
    ----------
        map_ : np.ndarray
            The input map to be partitioned. It should be a 1D array of pixel values.
        n_patches : int
            The number of patches to be created from the input map.
        mask : np.ndarray, optional
            A mask to be applied to the input map. If provided, only the pixels where the mask is non-zero will be considered for partitioning.
    
    Returns
    -------
        np.ndarray
            A 1D array of the same shape as the input map, where each pixel is assigned a patch number from 0 to n_patches-1.
    """
    if mask is None:
        mask = np.ones_like(map_)

    min_fraction = (0.05 * (np.sum(mask > 0.)/mask.shape[0]) )/ n_patches
    while True:
        partition = np.random.uniform(low=0.0, high=1.0, size=n_patches)
        partition = partition / np.sum(partition)
        if np.all(partition >= min_fraction):
            break

    partition = np.cumsum(partition)
    
    # Define index bins for partitioning the sorted map
#    total_pixels = map_.shape[0]
#    sorted_indices = np.argsort(map_)
#    bins = [0] + [int(total_pixels * partition[i]) for i in range(n_patches - 1)] + [total_pixels]

#    patches = np.zeros(12 * (hp.get_nside(map_))**2)
#    for i in range(n_patches):
#        patches[sorted_indices[bins[i]:bins[i+1]]]=i

    unmasked_indices = np.where(mask > 0.)[0]
    unmasked_map = map_[unmasked_indices]

    # Sort only the unmasked values
    sorted_unmasked_indices = unmasked_indices[np.argsort(unmasked_map)]

    total_unmasked = len(unmasked_indices)
    bins = [0] + [int(total_unmasked * partition[i]) for i in range(n_patches - 1)] + [total_unmasked]

    patches = np.zeros(12 * (hp.get_nside(map_))**2)

    # Assign patch indices only to unmasked pixels
    for i in range(n_patches):
        patch_indices = sorted_unmasked_indices[bins[i]:bins[i+1]]
        patches[patch_indices] = i

    return patches

def _adapt_tracers_path(
    path_tracers: Union[str, List[str]],
    n_fields: int = 1
) -> Union[str, List[str]]:
    """
    Adapt the path_tracers to the number of fields to be analysed.
    If n_fields is 1, it returns a single path as a string.
    If n_fields is greater than 1, it returns a list of paths with the same length as n_fields.

    Parameters
    ----------
        path_tracers : Union[str, List[str]]
            The path or list of paths to the tracers. It can be a single string or a list of strings.
        n_fields : int, optional
            The number of fields to be analysed. Default is 1.
        
    Returns
    -------
        Union[str, List[str]]
            The adapted path or list of paths to the tracers. If n_fields is 1, it returns a single string.
            If n_fields is greater than 1, it returns a list of strings with the same length as n_fields.
    """

    if n_fields == 1:
        if isinstance(path_tracers, str):
            return path_tracers
        elif isinstance(path_tracers, list):
            if len(path_tracers) == 1:
                return path_tracers[0]
            else:
                raise ValueError("If path_tracers is a list, it must contain only one element if you want to analyse just one field.")
    else:
        if isinstance(path_tracers, str):
            return [path_tracers] * n_fields
        elif isinstance(path_tracers, list):
            if len(path_tracers) == n_fields:
                return path_tracers
            else:
                raise ValueError("If path_tracers is a list, it must contain as many elements as the number of fields to be analysed.")

def get_tracers_compsep(
    channels_tracers: List[int],
    lmax: int
) -> List[Dict[str, Any]]:
    """
    Configure component separation methods for obtaining scalar tracers using GILC and needlet domains.

    Parameters
    ----------
        channels_tracers : List[int]
            List of channels indexes associated to the foreground tracers to be used in the component separation.
        lmax : int
            Maximum multipole for the component separation and analysis.
    
    Returns
    -------
        List[Dict[str, Any]]
            List of dictionaries containing the configuration for GILC run to obtain the scalar tracers.
    """    
    def generate_merging_needlets(starting_list: List[int]) -> List[int]:
        needlet_config = {
            "needlet_windows": "mexican",
            "width": 1.3,
            "merging_needlets": starting_list.copy(),
        }
        bl = _get_needlet_windows_(needlet_config, lmax)
        while np.abs(np.sum(bl ** 2, axis=0)[-1] - 1.0) > 1e-5:
            needlet_config["merging_needlets"].append(needlet_config["merging_needlets"][-1] + 5)
            bl = _get_needlet_windows_(needlet_config, lmax)
        return needlet_config["merging_needlets"]

    merging_needlets_1 = generate_merging_needlets([0, 16, 20, 23, 25, 27, 29, 40])
    merging_needlets_2 = generate_merging_needlets([0, 12, 15, 20, 23, 25, 27, 29, 40])


    tracers_compsep = [
    {"method": "gilc",
    "domain": "needlet",
    "ilc_bias": 0.13,
    "needlet_config":[
    {"needlet_windows": "mexican",
    "width": 1.3,
    "merging_needlets": merging_needlets_1}],
    "channels_out": channels_tracers,
    },
    {"method": "gilc",
    "domain": "needlet",
    "ilc_bias": 0.13,
    "needlet_config":[
    {"needlet_windows": "mexican",
    "width": 1.3,
    "merging_needlets": merging_needlets_2}],
    "channels_out": channels_tracers,
    }]
    return tracers_compsep

def get_mc_config(config: Configs, tracers_inputs_path: str) -> Configs:
    """
    Generate a configuration object for the MC-ILC tracers generation based on the provided configuration.

    Parameters
    ----------
    config : Configs
            Configuration object containing the instrumental and parameters configuration. 
            See 'generate_and_save_real_tracers_B' for details.
        tracers_inputs_path : str
            Path where the MC-ILC tracers inputs are stored. It should be a directory path.

    Returns
    -------
        Configs
            A new configuration object for the MC-ILC tracers generation, with paths and parameters set according to the provided configuration.
    """

    config_mc = Configs(config=config.to_dict_for_mc())
    config_mc.return_fgd_components = False

#    if config_mc.bandpass_integrate:
#        config_mc.data_path = f"inputs_mc_tracers/{config_mc.experiment}/total/{''.join(config_mc.foreground_models)}/total_bp_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
#        config_mc.fgds_path = f"inputs_mc_tracers/{config_mc.experiment}/foregrounds/{''.join(config_mc.foreground_models)}/foregrounds_bp_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
#    else:
#        config_mc.data_path = f"inputs_mc_tracers/{config_mc.experiment}/total/{''.join(config_mc.foreground_models)}/total_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
#        config_mc.fgds_path = f"inputs_mc_tracers/{config_mc.experiment}/foregrounds/{''.join(config_mc.foreground_models)}/foregrounds_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
#    config_mc.noise_path = f"inputs_mc_tracers/{config_mc.experiment}/noise/noise_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
#    config_mc.cmb_path = f"inputs_mc_tracers/{config_mc.experiment}/cmb/cmb_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
    if config_mc.bandpass_integrate:
        config_mc.data_path = f"{tracers_inputs_path}/total_{''.join(config_mc.foreground_models)}_bp_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
        config_mc.fgds_path = f"{tracers_inputs_path}/foregrounds_{''.join(config_mc.foreground_models)}_bp_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
    else:
        config_mc.data_path = f"{tracers_inputs_path}/total_{''.join(config_mc.foreground_models)}_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
        config_mc.fgds_path = f"{tracers_inputs_path}/foregrounds_{''.join(config_mc.foreground_models)}_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
    config_mc.noise_path = f"{tracers_inputs_path}/noise_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
    config_mc.cmb_path = f"{tracers_inputs_path}/cmb_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
    
#    if not all(os.path.exists(p + ".npy") for p in [config_mc.data_path, config_mc.noise_path, config_mc.cmb_path]):
#        if not os.path.exists(config_mc.fgds_path + f"_{''.join(config_mc.foreground_models)}.npy"):
    config_mc.generate_input_foregrounds = True #
#        else:
#            config_mc.generate_input_foregrounds = False
    config_mc.generate_input_noise = True #
    config_mc.generate_input_cmb = True #
    config_mc.generate_input_data = True #
    config_mc.save_inputs = True # 
    config_mc.seed_cmb = None
    config_mc.seed_noise = None
#    else:
#        if not os.path.exists(config_mc.fgds_path + f"_{''.join(config_mc.foreground_models)}.npy"):
#            config_mc.generate_input_foregrounds = True
#            config_mc.generate_input_data = True #
#            config_mc.save_inputs = True
#        else:
#            config_mc.generate_input_foregrounds = False
#            config_mc.generate_input_data = False
#            config_mc.save_inputs = False #
#       config_mc.generate_input_noise = False #
#       config_mc.generate_input_cmb = False #

    config_mc.save_compsep_products = False #
    config_mc.return_compsep_products = True
    config_mc.pixel_window_out = False
#    config_mc.mask_type = "mask_for_compsep"

    if config_mc.data_type == "maps":
        config_mc.field_in = "QU"
        config_mc.mc_data_field = "TQU"
    elif config_mc.data_type == "alms":
        config_mc.field_in = "B"
        config_mc.mc_data_field = "TEB"
    config_mc.field_out = "B"

    return config_mc

def get_mc_data(
    config_mc: "Configs",
    foregrounds: Optional[np.ndarray] = None,
    systematics: Optional[np.ndarray] = None,
    **kwargs
) -> Any:
    """
    Simulate the input data for the MC-ILC tracers generation based on the provided configuration.

    Parameters
    ----------
        config_mc : Configs
            Configuration object for the MC-ILC tracers generation. See 'get_mc_config' for details.
        foregrounds : SimpleNamespace, optional
            Foregrounds data to be used in the simulations for tracer generation. If None, the foregrounds will be generated using the configuration parameters.
        kwargs : dict, optional
            Additional keyword arguments to be passed to healpy 'map2alm' function.
    
    Returns
    -------
        SimpleNamespace
            A SimpleNamespace object containing the simulated data for the MC-ILC tracers generation. 
            It will have the following attributes:
            - `total`: The total simulated data, which includes CMB, foregrounds, and noise.
            - `cmb`: The simulated CMB data.
            - `fgds`: The simulated foregrounds data.
            - `noise`: The simulated noise data.
    """

    if foregrounds is None:
        mc_foregrounds = _get_data_foregrounds_(config_mc)
    else:
        mc_foregrounds = SimpleNamespace()
        mc_foregrounds.total = foregrounds

    mc_data = _get_data_simulations_(config_mc, mc_foregrounds)

    if systematics is not None:
        mc_data.total = mc_data.total + systematics

    return _slice_data(mc_data, config_mc.mc_data_field, config_mc.field_in)

def _combine_B_tracers(tracers, coefficients=[0.7,0.3]):
    """
    Combine the scalar tracers using the provided coefficients.

    Parameters
    ----------
        tracers : np.ndarray
            The scalar tracers to be combined. It can be either a 3D array (multiple tracers) or a 2D array (single tracer).
            If 2D, it should have shape (n_channels, n_pixels).
        coefficients : List[float], optional
            List of coefficients to be used for combining the tracers. Default is [0.7, 0.3].
            The coefficients should sum to 1. If they do not, they will be normalized.
    
    Returns
    -------
        np.ndarray
            The combined scalar tracers. If `tracers` is a 2D array, it returns the tracers as is.
            If `tracers` is a 3D array, it returns the weighted sum of the tracers using the provided coefficients. 
    """

    if tracers.ndim == 2:
        return tracers
    elif tracers.ndim == 3:
        if len(coefficients) != tracers.shape[0]:
            raise ValueError("The number of coefficients must match the number of tracers.")
        coefficients = np.array(coefficients) / np.sum(coefficients)
        return np.einsum("i,ijk->jk", coefficients, tracers)

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]
