import os
import numpy as np
import healpy as hp
import pysm3
import pysm3.units as u
from astropy.io import fits
from types import SimpleNamespace
from typing import Optional, Union, List, Dict, Any
import warnings
from threadpoolctl import threadpool_limits
import sys
from .configurations import Configs
from .seds import _get_CMB_SED
from .routines import (
    _get_ell_filter,
    _get_beam_from_file,
    _map2alm_kwargs, _log, _get_bandwidths, get_prefix,
)

prefix_to_attr = {
                "d": "dust", "s": "synch", "a": "ame", "co": "co",
                "f": "freefree", "cib": "cib", "tsz": "tsz",
                "ksz": "ksz", "rg": "radio_galaxies"
            }

def get_input_data(
    config: Configs,
    foregrounds: Optional[SimpleNamespace] = None,
    nsim: Optional[Union[int, str]] = None,
    **kwargs: Any
    ) -> SimpleNamespace:
    """
    Load or generate input data for component separation including total coadded signal, CMB, noise and foregrounds.

    Parameters
    ----------
        config: Configs
            Configuration parameters. It should have the following attributes:
            - `generate_input_cmb`: Whether to generate CMB maps. If False, it will try to load them from `cmb_path`.
            - `cmb_path`: Path where saving or loading CMB maps.
            - 'cls_cmb_path': Path to the CMB power spectrum FITS file. Used if 'generate_input_cmb' is True.
            - 'seed_cmb': Seed for CMB generation (optional).
            - 'cls_cmb_new_ordered': Whether the new ordering of Cls is used in the CMB power spectrum FITS file.
            - `generate_input_noise`: Whether to generate noise maps. If False, it will try load them from `noise_path`.
            - `noise_path`: Path where saving or loading noise maps.
            - `seed_noise`: Seed for noise generation (optional).
            - `data_splits`: Whether to generate/load noise and data splits.
            - `only_splits`: Whether to generate/load only noise and data splits, without full coadded maps.
            - `generate_input_data`: Whether to generate total data maps. If False, it will try to load them from `data_path`.
            - `data_path`: Path where saving or loading total data maps.
            - `save_inputs`: Whether to save generated inputs to disk.
            - `lmax_in`: Desired maximum multipole for the simulation.
            - `nside_in`: Desired HEALPix resolution. It will be used also to convolve the input maps for the pixel window function, if requested.
            - `units`: Units for the maps (e.g., 'uK_CMB').
            - `lmin_in`: Desired minimum multipole to keep in the simulation. Default is 2.
            - `pixel_window_in`: Whether to apply pixel window smoothing to the input maps.
            - 'generate_input_foregrounds': Whether to generate foreground maps. If False, it will try to load them from `fgds_path`.
            - `return_fgd_components`: Whether to return individual foreground components.
            - `fgds_path`: Path where saving or loading foreground maps.
            - `data_type`: Type of data to return, either "maps" or "alms".
            - `bandpass_integrate`: Whether to integrate sky components across bandpasses.
            - `coordinates`: Coordinate system for the maps (e.g., "G" for Galactic).
            - 'instrument': a dictionary containing the instrument configuration, including:
                - `frequency`: List of instrument frequencies in GHz.
                - `beams`: Type of beams to be used (e.g., "gaussian", "file_l", "file_lm").
                - `fwhm`: List of full width at half maximum (FWHM) for each frequency channel in arcmin. Used if beams are "gaussian".
                - 'depth_I': Depth for intensity maps in arcmin*uK_CMB (optional). 
                            If not provided, it will be assumed to be the polarization depth divided by sqrt(2).
                            Used if path_depth_maps is not provided.
                - 'depth_P': Depth for polarization maps in arcmin*uK_CMB (optional).
                            If not provided, it will be assumed to be the intensity depth multiplied by sqrt(2).
                            Used if path_depth_maps is not provided.
                - `path_beams`: Path to the beam files (if using "file_l" or "file_lm" beams).
                            The code will look for files named "{path_beams}_{channel_tag}.fits" for each frequency channel.
                - `channels_tags`: List of tags for each frequency channel, used for loading beams, bandpasses or depth maps.
                - 'bandwidths': List of relative bandwidths for each frequency channel (optional, used if bandpass_integrate is True).
                            Used if path_bandpasses is not provided.
                - `path_depth_maps`: Full path to standard deviation maps (optional, used if generating noise).
                            The code will look for files named "{path_depth_maps}_{channel_tag}.fits" for each frequency channel.   
                            They are assumed to be in uK_CMB units.
                - `path_hits_maps`: Full path to hits maps (optional, used if generating noise and 'path_depth_maps' is not provided).
                            If it does not end with .fits, the code will look for files named 
                            "{path_hits_maps}_{channel_tag}.fits" for each frequency channel.
                - `path_bandpasses`: Path to bandpass files (optional, used if bandpass_integrate is True).
                            The code will look for files named as "{path_bandpasses}_{channel_tag}.npy" for each channel tag.
                            Each file should be a 2D array which has the first column a list of frequencies in GHz and the second column the corresponding bandpass response.
                - `ell_knee`: Lists of knee frequencies for each channel for the noise power spectrum (optional).
                            If it is a single list it will be applied to temperature only.
                            If it is a list of two lists it will be applied to temperature (first list) and polarization (second list).
                            If not provided, white noise is assumed.
                - `alpha_knee`: List of spectral indices of the noise power spectrum for each channel (optional).
                            If not provided, white noise is assumed.
        foregrounds: Optional[SimpleNamespace]
            Foreground components. If provided, they will be used instead of generating or loading them.
        nsim: Optional[Union[int, str]]
            Simulation number.
        kwargs: dict, optional
            Additional keyword arguments forwarded to alm computation.

    Returns
    -------
        SimpleNamespace
            Data container potentially including co-added signal, CMB, noise, and foregrounds.
    """
    kwargs = _map2alm_kwargs(**kwargs)
    
    data = SimpleNamespace()

    if foregrounds is None:
        if config.generate_input_foregrounds or (config.fgds_path is not None): 
            foregrounds = _get_foregrounds_(config, **kwargs)
    
    if nsim is not None:
        if not isinstance(nsim, (int, str)):
            raise ValueError("nsim must be an integer or a string.")
        if isinstance(nsim, int):
            nsim = str(nsim).zfill(5)
    
    if foregrounds is not None: 
        if not hasattr(foregrounds, 'total'):
            raise ValueError('foregrounds must have the attribute total.')
        else:
            for attr, value in vars(foregrounds).items():
                if attr == 'total':
                    setattr(data, 'fgds', value)
                else:
                    setattr(data, attr, value)

    if config.generate_input_cmb or (config.cmb_path is not None):
        data.cmb = _get_cmb_(config, nsim=nsim)

    if config.generate_input_noise or (config.noise_path is not None):
        if not config.data_splits:
            data.noise = _get_noise_(config, nsim=nsim, **kwargs)
        else:
            if config.only_splits:
                data.noise_split1, data.noise_split2 = _get_noise_(config, nsim=nsim, **kwargs)
            else:
                data.noise, data.noise_split1, data.noise_split2 = _get_noise_(config, nsim=nsim, **kwargs)

    if config.generate_input_data:
        _log(f"Generating coadded signal" + f" for simulation {nsim}" if nsim is not None else "", verbose=config.verbose)
        if not config.data_splits or not config.only_splits:
            attrs_in = ["cmb", "noise", "fgds"]
            for attr in attrs_in:
                if hasattr(data, attr):
                    if not hasattr(data, 'total'):
                        data.total = np.copy(getattr(data, attr))
                    else:
                        data.total += getattr(data, attr)
            if not hasattr(data, 'total'):
                raise ValueError("To generate total input data, provide foregrounds, CMB or noise paths or ask to generate any of them.")
            if config.save_inputs:
                _save_inputs(config.data_path, data.total, nsim=nsim)
            
        if config.data_splits:
            attrs_in = ["cmb", "noise_split1", "fgds"]

            for attr in attrs_in:
                if hasattr(data, attr):
                    if not hasattr(data, 'total_split1'):
                        data.total_split1 = np.copy(getattr(data, attr))
                    else:
                        data.total_split1 += getattr(data, attr)
            if not hasattr(data, 'total_split1'):
                raise ValueError("To generate total input data splits, provide foregrounds, CMB or noise split paths or ask to generate any of them.")
            if config.save_inputs:
                _save_inputs(config.data_path + "_split1", data.total_split1, nsim=nsim)
            
            attrs_in = ["cmb", "noise_split2", "fgds"]
            for attr in attrs_in:
                if hasattr(data, attr):
                    if not hasattr(data, 'total_split2'):
                        data.total_split2 = np.copy(getattr(data, attr))
                    else:
                        data.total_split2 += getattr(data, attr)
            if not hasattr(data, 'total_split2'):
                raise ValueError("To generate total input data splits, provide foregrounds, CMB or noise split paths or ask to generate any of them.")
            if config.save_inputs:
                _save_inputs(config.data_path + "_split2", data.total_split2, nsim=nsim)
            
    elif config.data_path is not None:
        if not config.data_splits or not config.only_splits:
            data.total = _load_inputs(config.data_path, nsim=nsim)
        if config.data_splits:
            data.total_split1 = _load_inputs(config.data_path + "_split1", nsim=nsim)
            data.total_split2 = _load_inputs(config.data_path + "_split2", nsim=nsim)  

    return data

def _get_foregrounds_(config: Configs, **kwargs: Any) -> SimpleNamespace:
    """
    Load or generate foreground maps based on configuration.

    Parameters
    ----------
        config: Configs
            Configuration parameters. It should have the following attributes:
            - `generate_input_foregrounds`: Whether to generate foreground maps.
            - `foreground_models`: List of foreground models to generate.
            - 'instrument': a dictionary containing the instrument configuration, including:
                - `frequency`: List of instrument frequencies in GHz.
                - `beams`: Type of beams to be used (e.g., "gaussian", "file_l", "file_lm").
                - `fwhm`: List of full width at half maximum (FWHM) for each frequency channel in arcmin. Used if beams are "gaussian".
                - `path_beams`: Full path to the beams files (if using "file_l" or "file_lm" beams). 
                            The code will look for files named "{path_beams}_{channel_tag}.fits" for each frequency channel.
                - `channels_tags`: List of tags for each frequency channel, used for loading beams, bandpasses or depth maps.
                - 'bandwidths': List of relative bandwidths for each frequency channel (optional, used if bandpass_integrate is True).
                            Used if path_bandpasses is not provided.
                - `path_bandpasses`: Full path to bandpass files (optional, used if bandpass_integrate is True).
                            It will look for files named as "{path_bandpasses}_{channel_tag}.npy" for each channel tag.
                            Each file should be a 2D array which has the first column a list of frequencies in GHz and the second column the corresponding bandpass response.
            - `nside_in`: Desired HEALPix resolution.
            - `lmax_in`: Maximum multipole to keep in the simulation.
            - `return_fgd_components`: Whether to return individual foreground components.
            - `fgds_path`: Path where saving or loading foreground maps.
            - `save_inputs`: Whether to save generated foreground maps to disk.        
            - `pixel_window_in`: Whether to apply pixel window smoothing.
            - `units`: Units for the foreground maps (e.g., 'uK_CMB').
            - `data_type`: Type of data to return, either "maps" or "alms".
            - `bandpass_integrate`: Whether to integrate foreground components across bandpasses.
            - `lmin_in`: Minimum multipole to keep in the simulation.
            - `coordinates`: Coordinate system for the maps (e.g., "G" for Galactic).
        kwargs: dict, optional
            Additional keyword arguments forwarded to alm computation.

    Returns
    -------
        SimpleNamespace
            Foregrounds object containing single components (optionally) and total map.
    """
    kwargs = _map2alm_kwargs(**kwargs)

    if config.generate_input_foregrounds:
        if config.verbose:
            msg = f"Generating foreground maps of {''.join(config.foreground_models)} model"
            if config.bandpass_integrate:
                msg += " with bandpass integration"
            print(msg)

        foregrounds = _get_foregrounds_simulation(
            config.foreground_models,
            config.instrument,
            config.nside_in,
            config.lmax_in,
            return_components=config.return_fgd_components,
            pixel_window=config.pixel_window_in,
            units=config.units,
            return_alms=(config.data_type == "alms"),
            bandpass_integrate=config.bandpass_integrate,
            lmin=config.lmin_in,
            coordinates=config.coordinates,
            **kwargs
        )
        
        if config.save_inputs:
            _log(f"Saving foreground maps in {config.fgds_path} directory", verbose=config.verbose)
            _save_input_foregrounds(config.fgds_path, foregrounds, config.foreground_models)
    else:
        foregrounds = SimpleNamespace()
        if config.return_fgd_components:
            for fmodel in config.foreground_models:
                attr = prefix_to_attr.get(fmodel[:3]) or prefix_to_attr.get(fmodel[:2]) or prefix_to_attr.get(fmodel[:1])
                setattr(foregrounds, attr, _load_input_foregrounds(config.fgds_path, fmodel))
        foregrounds.total = _load_input_foregrounds(config.fgds_path, "".join(config.foreground_models))
    return foregrounds

def _get_cmb_(config: Configs, nsim: Optional[Union[int, str]] = None) -> np.ndarray:
    """
    Load or generate CMB maps based on configuration.

    Parameters
    ----------
        config: Configs
            Configuration parameters. It should have the following attributes:
            - `generate_input_cmb`: Whether to generate CMB maps.
            - `cmb_path`: Path where saving or loading CMB maps.
            - 'cls_cmb_path': Path to the CMB power spectrum FITS file. Used if 'generate_input_cmb' is True.
            - 'seed_cmb': Seed for CMB generation (optional).
            - 'cls_cmb_new_ordered': Whether the new ordering of Cls is used in the CMB power spectrum FITS file.
            - 'verbose': Whether to print progress messages.
        nsim: Optional[Union[int, str]]
            Simulation number.

    Returns
    -------
        np.ndarray
            CMB maps or alms. Shape is (n_freq, 3, n_pix) for maps or (n_freq, 3, n_alm) for alms.
    """

    if config.generate_input_cmb:
        _log(f"Generating CMB simulation" + f" {nsim}" if nsim is not None else "", verbose=config.verbose)
        return _get_cmb_simulation(config, nsim=nsim)
    elif config.cmb_path is not None:
        if config.verbose:
            path_str = f"{config.cmb_path}.npy" if nsim is None else f"{config.cmb_path}_{nsim}.npy"
            print(f"Loading CMB from {path_str}")
        return _load_inputs(config.cmb_path, nsim=nsim)

def _get_noise_(config: Configs, nsim: Optional[Union[int, str]] = None, **kwargs: Any) -> np.ndarray:
    """
    Load or generate noise maps based on configuration.

    Parameters
    ----------
        config: Configs
            Configuration parameters. It should have the following attributes:
            - `generate_input_noise`: Whether to generate noise maps.
            - `noise_path`: Path where saving or loading noise maps.
            - `seed_noise`: Seed for noise generation (optional).
            - 'verbose': Whether to print progress messages.
        nsim: Optional[Union[int, str]]
            Simulation number.
        kwargs: dict, optional
            Additional keyword arguments forwarded to alm computation.
    
    Returns
    -------
        np.ndarray
            Noise maps or alms. Shape is (n_freq, 3, n_pix) for maps or (n_freq, 3, n_alm) for alms.
    """

    if config.generate_input_noise:
        _log(f"Generating noise simulation" + f" {nsim}" if nsim is not None else "", verbose=config.verbose)
        if not config.data_splits:
            return _get_noise_simulation(config, nsim=nsim, **kwargs)
        else:
            noise_splits = _get_noise_simulation(config, nsim=nsim, **kwargs)
            if config.only_splits:
                return noise_splits[:,0], noise_splits[:,1]
            else:
                if config.save_inputs:
                    _save_inputs(config.noise_path, 0.5 * np.sum(noise_splits, axis=1), nsim=nsim)
                return 0.5 * np.sum(noise_splits, axis=1), noise_splits[:,0], noise_splits[:,1]
    elif config.noise_path is not None:
        if not config.data_splits:
            if config.verbose:
                path_str = f"{config.noise_path}.npy" if nsim is None else f"{config.noise_path}_{nsim}.npy"
                print(f"Loading noise from {path_str}")
            return _load_inputs(config.noise_path, nsim=nsim)
        else:
            if config.verbose:
                path_str1 = f"{config.noise_path}_split1.npy" if nsim is None else f"{config.noise_path}_split1_{nsim}.npy"
                path_str2 = f"{config.noise_path}_split2.npy" if nsim is None else f"{config.noise_path}_split2_{nsim}.npy"
                print(f"Loading noise split 1 from {path_str1}")
                print(f"Loading noise split 2 from {path_str2}")
                if not config.only_splits:
                    path_str = f"{config.noise_path}.npy" if nsim is None else f"{config.noise_path}_{nsim}.npy"
                    if os.path.exists(path_str):
                        print(f"Also loading coadded noise from {path_str}")
                    else:
                        print(f"Coadded noise file {path_str} not found. It will be computed as average of splits.")
            noise_split1 = _load_inputs(config.noise_path + "_split1", nsim=nsim)
            noise_split2 = _load_inputs(config.noise_path + "_split2", nsim=nsim)
            if not config.only_splits:
                coadded_path = config.noise_path + (f"_{nsim}" if nsim is not None else "") + ".npy"
                if os.path.exists(coadded_path):
                    noise = _load_inputs(config.noise_path, nsim=nsim)
                    return noise, noise_split1, noise_split2
                if config.save_inputs:
                    _save_inputs(config.noise_path, 0.5 * (noise_split1 + noise_split2), nsim=nsim)
                return 0.5 * (noise_split1 + noise_split2), noise_split1, noise_split2
            return noise_split1, noise_split2

def _save_inputs(filename: str, maps: np.ndarray, nsim: Union[str, None] = None) -> None:
    """Save simulation maps to disk, creating directories if needed.
    
    Parameters
    ----------
        filename: str
            Path to save the simulation maps, without extension.
        maps: np.ndarray
            Simulation maps to save. Shape is (n_freq, 3, n_pix) for maps or (n_freq, 3, n_alm) for alms.
        nsim: Union[str, None], optional
            Simulation index to append to the filename (optional). Default is None.
        
    Returns
    -------
        None
    
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if nsim is not None:
        filename += f"_{nsim}"
    np.save(filename, maps)

def _load_inputs(path: str, nsim: Union[str, None] = None) -> np.ndarray:
    """
    Load simulation maps from disk, handling nsim suffix.
    
    Parameters
    ----------
        path: str
            Path to the simulation maps file, without extension.
        nsim: Union[str, None], optional
            Simulation index to append to the filename (optional). Default is None.
    
    Returns
    -------
        np.ndarray
            Loaded simulation maps. Shape is (n_freq, 3, n_pix) for maps or (n_freq, 3, n_alm) for alms.
    """

    filepath = path + f"_{nsim}.npy" if nsim is not None else path + '.npy'
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return np.load(filepath)
    
def _save_input_foregrounds(fgds_path: str, foregrounds: SimpleNamespace, foreground_models: List[str]) -> None:
    """
    Save foreground maps for all models and the total.
    Assumes 'foregrounds' is a SimpleNamespace or similar with components as attributes.

    Parameters
    ----------
        fgds_path: str
            Path where saving foreground maps, without extension.
        foregrounds: SimpleNamespace
            Foreground components, should have attributes for each model and a 'total' attribute.
        foreground_models: List[str]
            List of foreground model names to save. Each name should match an attribute in 'foregrounds'.

    Returns
    -------
        None    
    """
    os.makedirs(os.path.dirname(fgds_path), exist_ok=True)
    # Save total foreground map
    np.save(fgds_path + f'_{"".join(foreground_models)}', foregrounds.total)

    if len(vars(foregrounds)) > 1:
        fg_attrs = {k: v for k, v in vars(foregrounds).items() if k != "total"}
        if len(fg_attrs) != len(foreground_models):
            raise ValueError(
                f"Number of foreground components ({len(fg_attrs)}) does not match number of models ({len(foreground_models)}).")

        for fmodel in foreground_models:
            # Try matching longest prefix first (3,2,1)
            attr = None
            for length in (3, 2, 1):
                if len(fmodel) > length:
                    prefix = fmodel[:length]
                    attr = prefix_to_attr.get(prefix)
                    if attr is not None:
                        break
            if attr is None:
                raise ValueError(f"Unknown foreground model prefix for '{fmodel}'")
            if attr not in fg_attrs:
                raise ValueError(f"Foreground attribute '{attr}' missing in foregrounds object")

            np.save(fgds_path + f"_{fmodel}", fg_attrs[attr])

def _load_input_foregrounds(fgd_path: str, fgd_model: str) -> np.ndarray:
    """
    Load foreground map for a given model.
    
    Parameters
    ----------
        fgd_path: str
            Path to the foreground maps, without extension.
        fgd_model: str
            Foreground model name to load. It should match the saved file name suffix.

    Returns
        np.ndarray
            Loaded foreground map. Shape is (n_freq, 3, n_pix) for maps or (n_freq, 3, n_alm) for alms.
    """
    filepath = f'{fgd_path}_{fgd_model}.npy'
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Foreground file not found: {filepath}")
    return np.load(filepath)

def _get_noise_simulation(config: Configs, nsim: Optional[Union[int, str]] = None, **kwargs) -> np.ndarray:
    """
    Generate noise simulations for the instrument frequencies.

    Parameters
    ----------
        config: Configs
            Configuration parameters including instrument settings. It should have the following attributes:
            - `instrument.frequency`: List of instrument frequencies.
            - `instrument.depth_I`: Depth for intensity maps.
            - `instrument.depth_P`: Depth for polarization maps.
            - `instrument.path_depth_maps`: Path to depth maps (optional). 
            - `instrument.path_hits_maps`: Path to hits maps (optional).
            - `nside_in`: Desired HEALPix resolution.
            - `lmax_in`: Maximum multipole for the simulation.
            - `data_type`: Type of data to return, either "maps" or "alms".
            - `units`: Units for the noise maps (e.g., 'uK_CMB').
            - `lmin_in`: Minimum multipole to keep in the simulation.
            - `seed_noise`: Seed for noise generation (optional).    
        nsim: int or str, optional
            Simulation index to save the maps and vary the random seed (optional). Default: None.
        kwargs: dict, optional
            Additional keyword arguments for `hp.map2alm`.

    Returns
    -------
        noise: np.ndarray
            array of noise maps or alms. Shape is (n_freq, 3, n_pix) for maps or (n_freq, 3, n_alm) for alms.
    """

    if nsim is not None:
        if not isinstance(nsim, (int, str)):
            raise ValueError("nsim must be an integer or a string.")
        if isinstance(nsim, int):
            nsim = str(nsim).zfill(5)

    # Precompute conversion factor from arcmin to radians
    acm_to_rad = (np.pi / (180 * 60)) 

    # Setup seed for reproducibility
    if config.seed_noise is None:
        seed = None
    else:
        if nsim is not None:
            if config.data_splits:
                seed = config.seed_noise + int(nsim) * 3 * len(config.instrument.frequency) * 2            
            else:
                seed = config.seed_noise + int(nsim) * 3 * len(config.instrument.frequency)
        else:
            seed = config.seed_noise    

    if not hasattr(config.instrument, 'path_depth_maps'):
        if not hasattr(config.instrument, 'depth_I') and not hasattr(config.instrument, 'depth_P'):
            raise ValueError('Provided instrumental setting must have either depth_I or depth_P attributes.')
        elif not hasattr(config.instrument, 'depth_I') and hasattr(config.instrument, 'depth_P'):
            config.instrument.depth_I = config.instrument.depth_P / np.sqrt(2)
            _log('Warning: No intensity map depth provided. Assuming it to be the polarization one divided by sqrt(2).', verbose=config.verbose)
        elif not hasattr(config.instrument, 'depth_P') and hasattr(config.instrument, 'depth_I'):
            config.instrument.depth_P = config.instrument.depth_I * np.sqrt(2)
            _log('Warning: No polarization map depth provided. Assuming it to be the intensity one multiplied by sqrt(2).', verbose=config.verbose)
        depth_i = config.instrument.depth_I
        depth_p = config.instrument.depth_P

        # Load or set hits map if available
        if hasattr(config.instrument, 'path_hits_maps'):
            if config.instrument.path_hits_maps.endswith(".fits"):
                hits_map = hp.read_map(config.instrument.path_hits_maps, field=0, dtype=np.float64)
                if hp.get_nside(hits_map) != config.nside_in:
                    hits_map = hp.ud_grade(hits_map, nside_out=config.nside_in, power=-2)
                hits_map /= np.amax(hits_map)
                
    else:
        omega_pix = (4 * np.pi) / hp.nside2npix(config.nside_in)
        depth_i = [np.sqrt(omega_pix) / acm_to_rad] * len(config.instrument.frequency)
        depth_p = [np.sqrt(omega_pix) / acm_to_rad] * len(config.instrument.frequency)
        
    # Convert depths to requested units with CMB equivalencies
    #depth_i *= u.arcmin * u.uK_CMB
    #depth_i = depth_i.to(getattr(u, config.units) * u.arcmin, equivalencies=u.cmb_equivalencies(config.instrument.frequency * u.GHz))
    #depth_p *= u.arcmin * u.uK_CMB
    #depth_p = depth_p.to(getattr(u, config.units) * u.arcmin, equivalencies=u.cmb_equivalencies(config.instrument.frequency * u.GHz))
    bandwidths = _get_bandwidths(config, np.arange(len(config.instrument.frequency)))
    A_cmb = _get_CMB_SED(config.instrument.frequency, units=config.units, bandwidths=bandwidths)
    depth_i = depth_i / A_cmb
    depth_p = depth_p / A_cmb

    if config.data_splits:
        depth_i = depth_i * np.sqrt(2)
        depth_p = depth_p * np.sqrt(2)

    # Get ell filter if needed
    fell = _get_ell_filter(config.lmin_in, config.lmax_in) if config.lmin_in > 0 else None

    noise = []
    for nf, _ in enumerate(config.instrument.frequency):

        # Load depth maps if path provided
        if hasattr(config.instrument, 'path_depth_maps'):
            depth_map_fn = config.instrument.path_depth_maps + f"_{config.instrument.channels_tags[nf]}.fits"
            try:
                depth_maps_in = hp.read_map(depth_map_fn, field=(0,1), dtype=np.float64)
            except IndexError:
                print("Warning: Unable to read depth maps from the provided path for I and P, provided depth map is assumed to refer to polarization.")
                depth_maps_in = hp.read_map(depth_map_fn, field=0, dtype=np.float64)
                depth_maps_in = np.array([depth_maps_in / np.sqrt(2), depth_maps_in])
            if hp.get_nside(depth_maps_in[0]) != config.nside_in:
                depth_maps = np.array(
                    [np.sqrt(hp.ud_grade(dm**2, nside_out=config.nside_in, power=2)) for dm in depth_maps_in])
            else:
                depth_maps = np.copy(depth_maps_in)
            del depth_maps_in

        elif hasattr(config.instrument, 'path_hits_maps'):
            if not config.instrument.path_hits_maps.endswith(".fits"):
                hits_file = config.instrument.path_hits_maps + f"_{config.instrument.channels_tags[nf]}.fits"
                hits_map = hp.read_map(hits_file, field=0, dtype=np.float64)
                if hp.get_nside(hits_map) != config.nside_in:
                    hits_map = hp.ud_grade(hits_map, nside_out=config.nside_in, power=-2)
                hits_map /= np.amax(hits_map)

        
        if not config.data_splits:
            noise.append(get_noise_frequency_channel(
                config,
                depth_i,
                depth_p,
                nf,
                depth_maps=depth_maps if hasattr(config.instrument, 'path_depth_maps') else None,
                hits_map=hits_map if hasattr(config.instrument, 'path_hits_maps') and not hasattr(config.instrument, 'path_depth_maps') else None,
                seed=seed,
                fell=fell,
                **kwargs
            ))
        else:
            noise_split1 = get_noise_frequency_channel(
                config,
                depth_i,
                depth_p,
                nf,
                depth_maps=depth_maps if hasattr(config.instrument, 'path_depth_maps') else None,
                hits_map=hits_map if hasattr(config.instrument, 'path_hits_maps') and not hasattr(config.instrument, 'path_depth_maps') else None,
                seed=seed,
                fell=fell,
                split=1,
                **kwargs
            )
            noise_split2 = get_noise_frequency_channel(
                config,
                depth_i,
                depth_p,
                nf,
                depth_maps=depth_maps if hasattr(config.instrument, 'path_depth_maps') else None,
                hits_map=hits_map if hasattr(config.instrument, 'path_hits_maps') and not hasattr(config.instrument, 'path_depth_maps') else None,
                seed=seed,
                fell=fell,
                split=2,
                **kwargs
            )
            noise.append([noise_split1, noise_split2])

    if config.save_inputs:
        if not config.data_splits:
            _save_inputs(config.noise_path, np.array(noise), nsim=nsim)
        else:
            _save_inputs(config.noise_path + "_split1", np.array(noise)[:,0], nsim=nsim)
            _save_inputs(config.noise_path + "_split2", np.array(noise)[:,1], nsim=nsim)

    return np.array(noise)

def get_noise_frequency_channel(config: Configs, depth_i: List[float], depth_p: List[float], nf: int, depth_maps: Optional[np.ndarray] = None, hits_map: Optional[np.ndarray] = None, seed: Optional[int] = None, fell: Optional[np.ndarray] = None, split: Optional[int] = None, **kwargs: Any) -> np.ndarray:
    """
    Generate noise simulation for a single frequency channel.

    Parameters
    ----------
        config: Configs
            Configuration parameters including instrument settings. It should have the following attributes:
            - `instrument.frequency`: List of instrument frequencies.
            - `instrument.path_depth_maps`: Path to depth maps (optional). 
            - `instrument.path_hits_maps`: Path to hits maps (optional).
            - `nside_in`: Desired HEALPix resolution.
            - `lmax_in`: Maximum multipole for the simulation.
            - `data_type`: Type of data to return, either "maps" or "alms".
            - `units`: Units for the noise maps (e.g., 'uK_CMB').
            - `lmin_in`: Minimum multipole to keep in the simulation.
        depth_i: List[float]
            Depth for intensity maps for each frequency channel.
        depth_p: List[float]
            Depth for polarization maps for each frequency channel.
        nf: int
            Index of the frequency channel to generate noise for.
        depth_maps: Optional[np.ndarray]
            Depth maps to use for noise generation (optional). Default is None.
        hits_map: Optional[np.ndarray]
            Hits map to use for noise generation (optional). Default is None.
        seed: Optional[int]
            Seed for noise generation (optional). Default is None.
        fell: Optional[np.ndarray]
            Ell filter to apply (optional). Default is None.
        split: Optional[int]
            Data split index for noise generation (optional). Default is None.
        kwargs: dict, optional
            Additional keyword arguments for `hp.map2alm`.

    Returns
    -------
        np.ndarray
            Noise maps or alms for the specified frequency channel. Shape is (3, n_pix) for maps or (3, n_alm) for alms.

    """

    if seed is not None:
        if split is not None:
            if split == 1:
                np.random.seed(seed + (nf * 3))
            elif split == 2:
                np.random.seed(seed + (nf * 3) + (len(config.instrument.frequency) * 3))
        else:
            np.random.seed(seed + (nf * 3))
    # Generate noise power spectra
#        N_ell_T = (depth_i.value[nf] * acm_to_rad) ** 2 * np.ones(config.lmax + 1)
#        N_ell_P = (depth_p.value[nf] * acm_to_rad) ** 2 * np.ones(config.lmax + 1)
    acm_to_rad = (np.pi / (180 * 60)) 

    N_ell_T = ((depth_i[nf] * acm_to_rad) ** 2) * np.ones(config.lmax_in + 1)
    N_ell_P = ((depth_p[nf] * acm_to_rad) ** 2) * np.ones(config.lmax_in + 1)
    N_ell = np.array([N_ell_T, N_ell_P, N_ell_P, 0.*N_ell_P])

    # Add knee frequency noise if provided
    if hasattr(config.instrument, 'ell_knee') and hasattr(config.instrument, 'alpha_knee'):
        ell = np.arange(config.lmax_in + 1)
        if isinstance(config.instrument.alpha_knee, list) and isinstance(config.instrument.ell_knee, list):
            if np.array(config.instrument.alpha_knee).ndim == 2 and np.array(config.instrument.ell_knee).ndim == 2:
                if len(config.instrument.alpha_knee[0]) != len(config.instrument.ell_knee[0]) or len(config.instrument.alpha_knee[1]) != len(config.instrument.ell_knee[1]):
                    raise ValueError('alpha_knee and ell_knee must have the same length.')
                if (len(config.instrument.ell_knee[0]) != len(config.instrument.frequency)) or (len(config.instrument.ell_knee[1]) != len(config.instrument.frequency)):
                    raise ValueError('alpha_knee and ell_knee must have the same length as the number of frequencies.')
                N_ell[0] *= (1 + (ell / config.instrument.ell_knee[0][nf]) ** config.instrument.alpha_knee[0][nf])
                N_ell[1:] *= (1 + (ell / config.instrument.ell_knee[1][nf]) ** config.instrument.alpha_knee[1][nf])
            elif np.array(config.instrument.alpha_knee).ndim == 1 and np.array(config.instrument.ell_knee).ndim == 1:
                if len(config.instrument.alpha_knee) != len(config.instrument.frequency) or len(config.instrument.ell_knee) != len(config.instrument.frequency):
                    raise ValueError('alpha_knee and ell_knee must have the same length as the number of frequencies.')
                N_ell[0] *= (1 + (ell / config.instrument.ell_knee[nf]) ** config.instrument.alpha_knee[nf])
            elif np.array(config.instrument.alpha_knee).ndim == 1 and np.array(config.instrument.ell_knee).ndim == 2:
                if len(config.instrument.alpha_knee) != len(config.instrument.frequency):
                    raise ValueError('alpha_knee must have the same length as the number of frequencies.')
                if len(config.instrument.ell_knee[0]) != len(config.instrument.frequency) or len(config.instrument.ell_knee[1]) != len(config.instrument.frequency):
                    raise ValueError('ell_knee lists must have the same length as the number of frequencies.')
                N_ell[0] *= (1 + (ell / config.instrument.ell_knee[0][nf]) ** config.instrument.alpha_knee[nf])
                N_ell[1:] *= (1 + (ell / config.instrument.ell_knee[1][nf]) ** config.instrument.alpha_knee[nf])
        else:
            raise ValueError('alpha_knee and ell_knee must be both lists or lists of 2 lists')

    N_ell[:,0] = 0.
    # Generate noise alm
    alm_noise = hp.synalm(N_ell, lmax=config.lmax_in, new=True)

    # Apply ell filter if applicable
    if fell is not None:
        for f in range(3):
            alm_noise[f] = hp.almxfl(alm_noise[f], fell)

    # Generate noise maps or alms depending on data_type
    if config.data_type=="alms":
        if depth_maps is not None:
            noise_map = hp.alm2map(alm_noise, config.nside_in, lmax=config.lmax_in, pol=True)  * np.array([depth_maps[0], depth_maps[1], depth_maps[1]])
            return hp.map2alm(noise_map, lmax=config.lmax_in, pol=True, **kwargs)
        elif hits_map is not None:
            noise_map = hp.alm2map(alm_noise, config.nside_in, lmax=config.lmax_in, pol=True) / np.sqrt(hits_map)
            noise_map[np.isinf(noise_map)] = 0.
            return hp.map2alm(noise_map, lmax=config.lmax_in, pol=True, **kwargs)
        else:
            return alm_noise
    else:
        if depth_maps is not None:
            return hp.alm2map(alm_noise, config.nside_in, lmax=config.lmax_in, pol=True) * np.array([depth_maps[0], depth_maps[1], depth_maps[1]])
        elif hits_map is not None:
            noise_map = hp.alm2map(alm_noise, config.nside_in, lmax=config.lmax_in, pol=True) / np.sqrt(hits_map)
            noise_map[np.isinf(noise_map)] = 0.
            return noise_map
        else:
            return hp.alm2map(alm_noise, config.nside_in, lmax=config.lmax_in, pol=True)

def _get_cmb_simulation(config: Configs, nsim: Optional[Union[int, str]] = None) -> np.ndarray:
    """
    Generate simulated CMB maps or alms for a given instrument configuration.

    Parameters
    ----------
        config: Configs
            Simulation and instrument configuration. It should have the following attributes:
            - `lmax_in`: Maximum multipole for the simulation.
            - `lmin_in`: Minimum multipole to keep in the simulation.
            - `nside_in`: Desired HEALPix resolution.
            - `data_type`: Type of data to return, either "maps" or "alms".
            - `cls_cmb_path`: Path to the CMB power spectrum FITS file.
            - `seed_cmb`: Seed for CMB generation (optional).
            - 'cls_cmb_new_ordered': Whether the new ordering of Cls is used in the CMB power spectrum FITS file.
        nsim: int or str, optional
            Simulation index to save the maps and vary the random seed (optional). Default: None.

    Returns
    -------
        np.ndarray: 
            Simulated CMB maps or harmonic coefficients (alms). Shape is (n_freq, 3, n_pix) for maps or (n_freq, 3, n_alm) for alms.
    """
    # Converting nsim to a string if provided
    if nsim is not None:
        if not isinstance(nsim, (int, str)):
            raise ValueError("nsim must be an integer or a string.")
        if isinstance(nsim, int):
            nsim = str(nsim).zfill(5)

    # Getting default path to CMB power spectrum, if not provided in config
    if not config.cls_cmb_path:
        # Define the path to the FITS file
        raise ValueError(
            "No CMB power spectrum path provided. Please set 'cls_cmb_path' in the configuration."
        )
        
    # Load the CMB power spectrum FITS file (assumed to be in muK_CMB^2 units)
    cls_cmb = hp.read_cl(config.cls_cmb_path)

    # Initializing the seed if required
    seed = None if not config.seed_cmb else (config.seed_cmb + 3 * int(nsim) if nsim is not None else config.seed_cmb)
    
    # Generating a realization of CMB alms with the loaded Cls
    alm_cmb = _get_cmb_alms_realization(cls_cmb, config.lmax_in, seed = seed, new = config.cls_cmb_new_ordered)
    
    # Computing the high-pass filter if lmin > 2
    fell = _get_ell_filter(config.lmin_in, config.lmax_in) if config.lmin_in > 0 else None

    # Smoothing the CMB alms with the beams of each frequency channel
    cmb = []

    bandwidths = _get_bandwidths(config, np.arange(len(config.instrument.frequency)))
    A_cmb = _get_CMB_SED(config.instrument.frequency, units=config.units, bandwidths=bandwidths)
     
    for idx, _ in enumerate(config.instrument.frequency):
        if config.instrument.beams == "gaussian":
            alm_cmb_i = _smooth_input_alms_(
                alm_cmb,
                fwhm=config.instrument.fwhm[idx],
                nside_out=config.nside_in if config.pixel_window_in else None
            )
        else:
            beamfile = config.instrument.path_beams + f"_{config.instrument.channels_tags[idx]}.fits"
            alm_cmb_i = _smooth_input_alms_(
                alm_cmb,
                beam_path=beamfile,
                symmetric_beam=(config.instrument.beams == "file_l"),
                nside_out=config.nside_in if config.pixel_window_in else None
            )

        if fell is not None:
            for f in range(3):
                alm_cmb_i[f] = hp.almxfl(alm_cmb_i[f], fell)

        cmb.append((alm_cmb_i / A_cmb[idx]) if config.data_type == "alms" else hp.alm2map(
            alm_cmb_i, config.nside_in, lmax=config.lmax_in, pol=True
        ) / A_cmb[idx])
    
    cmb = np.array(cmb)

    # Saving the CMB maps/alms if requested
    if config.save_inputs:
        _save_inputs(config.cmb_path, cmb, nsim=nsim)
    return cmb

def _get_cmb_alms_realization(
    cls_cmb: np.ndarray,
    lmax: int,
    seed: Optional[int] = None,
    new: bool = True
) -> np.ndarray:
    """
    Generate a realization of CMB spherical harmonic coefficients (alms).

    Parameters
    ----------
        cls_cmb: np.ndarray
            Theoretical CMB angular power spectra.
        lmax: int
            Maximum multipole for the realization.
        seed: Optional[int]
            Random seed for reproducibility.
        new: bool
            healpy sinalm keyword which sets the assumed ordering of the Cls. Default: True.
                    If True, use the new ordering of cl’s, ie by diagonal (e.g. TT, EE, BB, TE, EB, TB or TT, EE, BB, TE if 4 cl as input). 
                    If False, use the old ordering, ie by row (e.g. TT, TE, TB, EE, EB, BB or TT, TE, EE, BB if 4 cl as input).

    Returns
    -------
        np.ndarray
            Realization of CMB alms for T, E, and B modes. Shape is (3, n_alms)
    """
    if seed is not None:
        np.random.seed(seed)
    return hp.synalm(cls_cmb, lmax=lmax, new=new)

def _get_foregrounds_simulation(
    foreground_models: List[str],
    instrument: dict, 
    nside: int,
    lmax: int,
    return_components: bool = False,
    pixel_window: bool = False,
    units: str = 'uK_CMB',
    return_alms: bool = False,
    bandpass_integrate: bool = False,
    lmin: int = 2,
    coordinates: str = "G",
    **kwargs) -> SimpleNamespace:
    """
    Generate simulated foregrounds from PySM3 models for a given instrument.

    Parameters
    ----------
        foreground_models: (List[str])
            List of PySM3 model presets (e.g., ["d1", "s1"]).
        instrument: dict
            Instrument configuration object with frequency, beams, and optional bandpasses.
        nside: int
            Desired HEALPix resolution. Used also to apply pixel window function (if requested)
        lmax: int
            Maximum multipole to compute alms.
        return_components: bool, optional
            If True, return individual components instead of just the sum. Default: False.
        pixel_window: bool, optional
            Whether to apply pixel window smoothing. Default: False.
        units: str, optional
            Output units. Default: 'uK_CMB'.
        return_alms: bool, optional
            Whether to return alms instead of maps. Default: False.
        bandpass_integrate: bool, optional
            Whether to integrate foreground components across bandpasses. 
            Default: False (i.e. delta functions are assumed).
        lmin: int, optional
            Minimum multipole to keep (applies filtering).
        coordinates, str, optional
            Target coordinate system for output maps/alms ("G" (Galactic), "E" (Ecliptic), or "C" (Equatorial)). 
            Default: "G"
        **kwargs: Additional keyword arguments for `hp.map2alm`.

    Returns
    -------
        SimpleNamespace: 
            Foregrounds object with `.total` field and optionally individual components.
    """

    nside_ = max(nside, 512)
    
    # Foregrounds initialization
    foregrounds = SimpleNamespace()

    # Derivation of foreground components for all instrument frequencies
    if not return_components or len(foreground_models) == 1:
        sky = pysm3.Sky(nside=nside_, preset_strings=foreground_models, output_unit=getattr(u, units))
        foregrounds.total = _get_foreground_component(
            instrument, sky, nside, lmax,
            pixel_window=pixel_window,
            bandpass_integrate=bandpass_integrate,
            return_alms=return_alms,
            lmin=lmin,
            coordinates=coordinates,
            **kwargs
        )
    else:
        for fmodel in foreground_models:
            sky = pysm3.Sky(nside=nside_, preset_strings=[fmodel], output_unit=getattr(u, units))
            attr = prefix_to_attr.get(fmodel[:3]) or prefix_to_attr.get(fmodel[:2]) or prefix_to_attr.get(fmodel[:1])
            setattr(foregrounds, attr, _get_foreground_component(
                instrument, sky, nside, lmax,
                pixel_window=pixel_window,
                bandpass_integrate=bandpass_integrate,
                return_alms=return_alms,
                lmin=lmin,
                coordinates=coordinates,
                **kwargs
            ))
        foregrounds.total = sum(vars(foregrounds).values())
    return foregrounds

def _get_foreground_component(
    instrument: dict,
    sky: pysm3.Sky,
    nside_out: int,
    lmax: int,
    pixel_window: bool = False,
    bandpass_integrate: bool = False,
    return_alms: bool = False,
    lmin: int = 2,
    coordinates: str = "G",
    **kwargs
) -> np.ndarray:
    """
    Generate a foreground component for each frequency channel of the instrument.

    Parameters
    ----------
        instrument: dict
            Instrument configuration with frequencies, fwhms (or beam paths), bandpasses or bandwidths.
        sky: pysm3.Sky
            PySM3 sky model for the foreground.
        nside_out: int
            HEALPix resolution for the output. Used also to apply pixel window function (if requested)
        lmax: int
            Maximum multipole to compute alms.
        pixel_window: bool, optional
            Apply pixel window smoothing if True. Default: False.
        bandpass_integrate: bool, optional
            Integrate over bandpass if True. Default: False.
        return_alms: bool, optional
            Return alms if True, else return maps. Default: False.
        lmin: int, optional
            Minimum multipole to keep (applies filtering).
        coordinates: str, optional
            Coordinate system for output ("G" (Galactic), "E" (Ecliptic), or "C" (Equatorial)). 
            Default: "G"
        **kwargs: Additional arguments passed to `hp.map2alm`.

    Returns
    -------
        np.ndarray: 
            Array of foreground maps or alms with shape (n_channels, 3, npix) for maps or (n_channels, 3, nalms) for alms.
    """

    fg_component = []

    fell = _get_ell_filter(lmin, lmax) if lmin > 0 else None

    rot = hp.Rotator(coord=f"G{coordinates}") if coordinates != "G" else None
    
    # Getting foreground component for each frequency channel
    for idx, freq in enumerate(instrument.frequency):
        if bandpass_integrate:
            if hasattr(instrument, 'path_bandpasses'):
                # Reading bandpass from file
                bandpass_file = instrument.path_bandpasses + f"_{instrument.channels_tags[idx]}.npy"
                frequencies, bandpass_weights = np.load(bandpass_file)
                frequencies = frequencies * u.GHz
            else:
                # Create a top-hat bandpass
                freq_min = freq * (1 - ( instrument.bandwidth[idx] / 2 ))
                freq_max = freq * (1 + ( instrument.bandwidth[idx] / 2 ))
                steps = int(freq_max - freq_min + 1)
                frequencies = np.linspace(freq_min, freq_max, steps) * u.GHz
                bandpass_weights = np.ones(len(frequencies)) # The tophat is defined in intensity units (Jy/sr)
            with threadpool_limits(limits=1):
                emission = sky.get_emission(frequencies, bandpass_weights)
        else:
            with threadpool_limits(limits=1):
                emission = sky.get_emission(freq * u.GHz)

        # Computing alms of the foreground emission
        alm_emission = hp.map2alm(emission.value, lmax=lmax, pol=True, **kwargs)

        # Applying coordinate rotation if needed
        if coordinates != "G":
            rot.rotate_alm(alm_emission, inplace=True)
        
        # Smoothing the alms with the instrument beam
        if instrument.beams == "gaussian":
            alm_emission = _smooth_input_alms_(
                alm_emission,
                fwhm=instrument.fwhm[idx],
                nside_out=nside_out if pixel_window else None
            )
        else:
            beamfile = instrument.path_beams + f"_{instrument.channels_tags[idx]}.fits"
            alm_emission = _smooth_input_alms_(
                alm_emission,
                beam_path=beamfile,
                symmetric_beam=(instrument.beams == "file_l"),
                nside_out=nside_out if pixel_window else None
            )

        if lmin > 2:
            for f in range(3):
                alm_emission[f] = hp.almxfl(alm_emission[f], fell)
        fg_component.append(alm_emission if return_alms else hp.alm2map(alm_emission, nside_out, lmax=lmax, pol=True))
        
    return np.array(fg_component)

def get_nuisance_data(config: Configs, nuisance_comps, nuisance_path: str = None, nsim: Optional[Union[int, str]] = None) -> SimpleNamespace:
    """
    Get nuisance data for nuisance covariance estimation for a given simulation.

    Parameters
    ----------
        config: Configs
            Configuration parameters including instrument settings and paths.
        nuisance_comps: List[str]
            List of nuisance components to include.
        nuisance_path: str, optional
            Path to precomputed nuisance inputs. If None, inputs will be generated/saved.
        nsim: int or str, optional
            Simulation index to load/save the maps (optional). Default: None.
        
    Returns
    -------
        SimpleNamespace: 
            Nuisance data containing CMB and noise simulations.

    """

    generate_input_foregrounds = config.generate_input_foregrounds
    return_fgd_components = config.return_fgd_components
    foreground_models = config.foreground_models
    generate_input_cmb = config.generate_input_cmb
    if generate_input_cmb:
        seed_cmb = config.seed_cmb
    generate_input_noise = config.generate_input_noise
    if generate_input_noise:
        seed_noise = config.seed_noise
    generate_input_data = config.generate_input_data
    cmb_path = config.cmb_path
    noise_path = config.noise_path
    fgds_path = config.cmb_path
    data_path = config.noise_path
    if "cmb" in nuisance_comps:
        cmbname = f"cmb_{config.data_type}_ns{config.nside_in}_lmax{config.lmax_in}"
    if "noise" in nuisance_comps:
        noisename = f"noise_{config.data_type}_ns{config.nside_in}_lmax{config.lmax_in}"
    if any(x not in ["cmb", "noise"] for x in nuisance_comps):
        nuis_fgds = [x for x in nuisance_comps if x not in ["cmb", "noise"]]
        fgdsname = f"foregrounds_{config.data_type}_ns{config.nside_in}_lmax{config.lmax_in}"
    data_splits = config.data_splits
    only_splits = config.only_splits 
    
    config.generate_input_foregrounds = False
    config.return_fgd_components = True
    config.foreground_models = None
    config.generate_input_data = False
    config.generate_input_cmb = False
    config.generate_input_noise = False
    config.data_path = None
    config.fgds_path = None
    config.cmb_path = None
    config.noise_path = None
    config.seed_cmb = None
    config.seed_noise = None
    config.data_splits = False
    config.only_splits = False

    if any(x not in ["cmb", "noise"] for x in nuisance_comps):
        prefix_models = ["d", "s", "co", "a", "f", "tsz", "cib", "ksz", "rg"]
        prefix_to_model = {}
        for model in nuis_fgds:
            prefix = get_prefix(model, prefix_models)
            if prefix is None:
                raise ValueError(f"Unknown prefix in model: {model}")
            if prefix in prefix_to_model:
                raise ValueError(f"Error: more models with prefix '{prefix}' in nuisance components.")
            prefix_to_model[prefix] = model

    if nuisance_path is not None:
        if "cmb" in nuisance_comps:
            config.cmb_path = os.path.join(nuisance_path, "cmb", cmbname)
        if "noise" in nuisance_comps:
            config.noise_path = os.path.join(nuisance_path, "noise", noisename)
        if any(x not in ["cmb", "noise"] for x in nuisance_comps):
            config.fgds_path = os.path.join(nuisance_path, "foregrounds", fgdsname)
        nuisance_data = get_input_data(config, nsim=nsim)
    else:
        if "cmb" in nuisance_comps:
            config.generate_input_cmb = True
        if "noise" in nuisance_comps:
            config.generate_input_noise = True

        if not hasattr(config, "save_inputs"):
            remove_inputs = True
            config.save_inputs = False
        else:
            remove_inputs = False
        if config.save_inputs:
            if "cmb" in nuisance_comps:
                config.cmb_path = os.path.join(os.getcwd(), "nuisance_inputs", config.experiment, "cmb", cmbname)
            if "noise" in nuisance_comps:
                config.noise_path = os.path.join(os.getcwd(), "nuisance_inputs", config.experiment, "noise", noisename)

        if "cmb" in nuisance_comps or "noise" in nuisance_comps:
            nuisance_data = get_input_data(config, nsim=nsim)
        else:
            nuisance_data = SimpleNamespace()

        if any(x not in ["cmb", "noise"] for x in nuisance_comps):
            config.generate_input_cmb = False
            config.generate_input_noise = False
            config.noise_path = None
            config.cmb_path = None
            config.fgds_path = os.path.join(os.getcwd(), "nuisance_inputs", config.experiment, "foregrounds", fgdsname)
            for model in nuis_fgds:
                config.foreground_models = [model]
                config.generate_input_foregrounds = not os.path.exists(config.fgds_path + f"_{''.join(config.foreground_models)}.npy")
                nuis_fgds_data = get_input_data(config)
                attr = [name for name in vars(nuis_fgds_data) if name != 'fgds'][0]
                setattr(nuisance_data, model, getattr(nuis_fgds_data, attr))
            del nuis_fgds_data

    if hasattr(nuisance_data, 'fgds'):
        delattr(nuisance_data, 'fgds')
    if hasattr(nuisance_data, 'total'):
        delattr(nuisance_data, 'total')

    config.generate_input_foregrounds = generate_input_foregrounds
    config.return_fgd_components = return_fgd_components
    config.foreground_models = foreground_models
    config.generate_input_cmb = generate_input_cmb
    if config.generate_input_cmb:
        config.seed_cmb = seed_cmb
    config.generate_input_noise = generate_input_noise
    if config.generate_input_noise:
        config.seed_noise = seed_noise
    config.generate_input_data = generate_input_data
    config.cmb_path = cmb_path
    config.noise_path = noise_path
    config.fgds_path = fgds_path
    config.data_path = data_path
    config.data_splits = data_splits
    config.only_splits = only_splits

    if nuisance_path is None and remove_inputs:
        delattr(config, "save_inputs")
        
    return nuisance_data
    
def _smooth_input_alms_(
    alms: np.ndarray,
    fwhm: Optional[float] = None,
    nside_out: Optional[int] = None,
    beam_path: Optional[str] = None,
    symmetric_beam: bool = True
) -> np.ndarray:
    """
    Apply beam and pixel window smoothing to input alms.

    Parameters
    ----------
        alms: np.ndarray
            Array of spherical harmonic coefficients [T, E, B].
        fwhm: float, optional
            FWHM of Gaussian beam in arcmin. Required if `beam_path` is None.
        nside_out: int, optional
            HEALPix Nside for pixel window function. Used if not None.
        beam_path: str, optional
            Path to FITS file containing beam transfer functions.
        symmetric_beam: bool, optional
            Whether the beam from FITS file is symmetric (True) or not (False).

    Returns
    -------
        np.ndarray: 
            Smoothed alms.
    """
    lmax = hp.Alm.getlmax(alms.shape[1])

    # Beams transfer functions
    if beam_path is not None:
        bl_i = _get_beam_from_file(beam_path, lmax,symmetric_beam=symmetric_beam)
    elif fwhm is not None:
        symmetric_beam = True
        bl_i = hp.gauss_beam(np.radians(fwhm/60.), lmax = lmax, pol = True)
    else:
        raise ValueError("Either fwhm or beam_path must be provided.")

    bl_i = bl_i[:,:3]  # Take only T, E, B

    if nside_out:
        pw = hp.pixwin(nside_out, pol=True, lmax=lmax)
        pw = np.array([pw[0], pw[1], pw[1]])
        if symmetric_beam:
            bl_i = bl_i * pw.T
        else:
            for i in range(3):
                bl_i[:,i] = hp.almxfl(bl_i[:,i], pw[i])

    # Initializing smoothed alms
    alms_smoothed = np.zeros_like(alms)

    for i in range(3):
        alms_smoothed[i] = hp.almxfl(alms[i], bl_i[:,i]) if symmetric_beam else alms[i] * bl_i[:,i]

    return alms_smoothed

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]


