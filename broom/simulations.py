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
    _map2alm_kwargs, _log, _get_bandwidths,
)

prefix_to_attr = {
                "d": "dust", "s": "synch", "a": "ame", "co": "co",
                "f": "freefree", "cib": "cib", "tsz": "tsz",
                "ksz": "ksz", "rg": "radio_galaxies"
            }

def _get_full_simulations(config: Configs, nsim: Optional[Union[int, str]] = None, **kwargs: Any) -> SimpleNamespace:
    """
    Generate full simulations including foregrounds and data.

    Parameters
    ----------
        config: Configs
            Configuration parameters.
        nsim: Optional[Union[int, str]]
            Simulation number.
        kwargs: dict, optional
            Additional keyword arguments forwarded to alm computation in map2alm.

    Returns
    -------
        SimpleNamespace
            Simulated data container with foregrounds and total data.
    """
    kwargs = _map2alm_kwargs(**kwargs)

    foregrounds = _get_data_foregrounds_(config, **kwargs)

    data = _get_data_simulations_(config, foregrounds, nsim=nsim, **kwargs)

    return data

def _get_data_foregrounds_(config: Configs, **kwargs: Any) -> SimpleNamespace:
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
                - 'depth_I': Depth for intensity maps in arcmin*uK_CMB (optional). 
                            If not provided, it will be assumed to be the polarization depth divided by sqrt(2).
                            Used if path_depth_maps is not provided.
                - 'depth_P': Depth for polarization maps in arcmin*uK_CMB (optional).
                            If not provided, it will be assumed to be the intensity depth multiplied by sqrt(2).
                            Used if path_depth_maps is not provided.
                - `path_beams`: Full path to the beams files (if using "file_l" or "file_lm" beams). 
                            The code will look for files named "{path_beams}_{channel_tag}.fits" for each frequency channel.
                - `channels_tags`: List of tags for each frequency channel, used for loading beams, bandpasses or depth maps.
                - 'bandwidths': List of relative bandwidths for each frequency channel (optional, used if bandpass_integrate is True).
                            Used if path_bandpasses is not provided.
                - `path_depth_maps`: Full path to depth maps (optional, used if generating noise). 
                            The code will look for files named "{path_depth_maps}_{channel_tag}.fits" for each frequency channel.
                - `path_hits_maps`: Full path to hits maps (optional, used if generating noise and 'depth_maps' is not provided).
                            If it does not end with .fits, the code will look for files named "{path_hits_maps}_{channel_tag}.fits" for each frequency channel.
                - `path_bandpasses`: Full path to bandpass files (optional, used if bandpass_integrate is True).
                            It will look for files named as "{path_bandpasses}_{channel_tag}.npy" for each channel tag.
                            Each file should be a 2D array which has the first column a list of frequencies in GHz and the second column the corresponding bandpass response.
                - `ell_knee`: Lists of knee frequencies for each channel for the noise power spectrum (optional).
                            If it is a single list it will be applied to temperature only.
                            If it is a list of two lists it will be applied to temperature (first list) and polarization (second list).
                            If not provided, white noise is assumed.
                - `alpha_knee`: List of spectral indices of the noise power spectrum for each channel (optional).
                            If not provided, white noise is assumed.
            - `nside`: HEALPix resolution.
            - `lmax`: Maximum multipole for the simulation.
            - `return_fgd_components`: Whether to return individual foreground components.
            - `fgds_path`: Path where saving or loading foreground maps.
            - `save_inputs`: Whether to save generated foreground maps to disk.        
            - `pixel_window_in`: Whether to apply pixel window smoothing.
            - `units`: Units for the foreground maps (e.g., 'uK_CMB').
            - `data_type`: Type of data to return, either "maps" or "alms".
            - `bandpass_integrate`: Whether to integrate foreground components across bandpasses.
            - `lmin`: Minimum multipole to keep in the simulation.
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

        foregrounds = _get_foregrounds(
            config.foreground_models,
            config.instrument,
            config.nside,
            config.nside_in,
            config.lmax,
            return_components=config.return_fgd_components,
            pixel_window=config.pixel_window_in,
            units=config.units,
            return_alms=(config.data_type == "alms"),
            bandpass_integrate=config.bandpass_integrate,
            lmin=config.lmin,
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


def _get_data_simulations_(
    config: Configs,
    foregrounds: Optional[SimpleNamespace] = None,
    nsim: Optional[Union[int, str]] = None,
    **kwargs: Any
    ) -> SimpleNamespace:
    """
    Load or generate simulation data including CMB, noise, and combined total.

    Parameters
    ----------
        config: Configs
            Configuration parameters. It should have the following attributes:
            - `generate_input_cmb`: Whether to generate CMB maps. If False, it will load from `cmb_path`.
            - `cmb_path`: Path where saving or loading CMB maps.
            - 'cls_cmb_path': Path to the CMB power spectrum FITS file. Used if 'generate_input_cmb' is True.
            - 'seed_cmb': Seed for CMB generation (optional).
            - 'cls_cmb_new_ordered': Whether the new ordering of Cls is used in the CMB power spectrum FITS file.
            - `generate_input_noise`: Whether to generate noise maps. If False, it will load from `noise_path`.
            - `noise_path`: Path where saving or loading noise maps.
            - `seed_noise`: Seed for noise generation (optional).
            - `generate_input_data`: Whether to generate total data maps. If False, it will load from `data_path`.
            - `data_path`: Path where saving or loading total data maps.
            - `save_inputs`: Whether to save generated inputs to disk.
            - `lmax`: Maximum multipole for the simulation.
            - `nside`: Desired HEALPix resolution.
            - `data_type`: Type of data to return, either "maps" or "alms". It must be compatible with provided foregrounds, if any.
            - `units`: Units for the maps (e.g., 'uK_CMB').
            - `lmin`: Minimum multipole to keep in the simulation. Default is 2.
            - `pixel_window_in`: Whether to apply pixel window smoothing to the input maps.
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
                - `path_depth_maps`: Full path to depth maps (optional, used if generating noise).
                            The code will look for files named "{path_depth_maps}_{channel_tag}.fits" for each frequency channel.   
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
            Foreground components.
        nsim: Optional[Union[int, str]]
            Simulation number.
        kwargs: dict, optional
            Additional keyword arguments forwarded to alm computation.

    Returns
    -------
        SimpleNamespace
            Data container with cmb, noise, total and foregrounds.
    """
    if nsim is not None:
        if not isinstance(nsim, (int, str)):
            raise ValueError("nsim must be an integer or a string.")
        if isinstance(nsim, int):
            nsim = str(nsim).zfill(5)
    
    kwargs = _map2alm_kwargs(**kwargs)

    if foregrounds is not None and not hasattr(foregrounds, 'total'):
        raise ValueError('foregrounds must have the attribute total.')

    data = SimpleNamespace()

    if config.generate_input_cmb:
        _log(f"Generating CMB simulation" + f" {nsim}" if nsim is not None else "", verbose=config.verbose)
        data.cmb = _get_cmb_simulation(config, nsim=nsim)
    elif config.cmb_path is not None:
        if config.verbose:
            path_str = f"{config.cmb_path}.npy" if nsim is None else f"{config.cmb_path}_{nsim}.npy"
            print(f"Loading CMB from {path_str}")
        data.cmb = _load_inputs(config.cmb_path, nsim=nsim)

    if config.generate_input_noise:
        _log(f"Generating noise simulation" + f" {nsim}" if nsim is not None else "", verbose=config.verbose)
        data.noise = _get_noise_simulation(config, nsim=nsim, **kwargs)
    elif config.noise_path is not None:
        if config.verbose:
            path_str = f"{config.noise_path}.npy" if nsim is None else f"{config.noise_path}_{nsim}.npy"
            print(f"Loading noise from {path_str}")
        data.noise = _load_inputs(config.noise_path, nsim=nsim)

    if config.generate_input_data:
        _log(f"Generating coadded signal" + f" for simulation {nsim}" if nsim is not None else "", verbose=config.verbose)
        if hasattr(data, 'cmb') and hasattr(data, 'noise') and foregrounds is not None:
            data.total = data.noise + data.cmb + foregrounds.total
            if config.save_inputs:
                _save_inputs(config.data_path, data.total, nsim=nsim)
        else:
            raise ValueError("To generate input data, provide foregrounds and CMB/noise paths or generate them.")
    else:
        data.total = _load_inputs(config.data_path, nsim=nsim)

    if foregrounds is not None:
        data.fgds = foregrounds.total

    return data

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
            - `nside`: HEALPix resolution.
            - `lmax`: Maximum multipole for the simulation.
            - `data_type`: Type of data to return, either "maps" or "alms".
            - `units`: Units for the noise maps (e.g., 'uK_CMB').
            - `lmin`: Minimum multipole to keep in the simulation.
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


    # Setup seed for reproducibility
    if config.seed_noise is None:
        seed = None
    else:
        if nsim is not None:
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
                if hp.get_nside(hits_map) != config.nside:
                    hits_map = hp.ud_grade(hits_map, nside_out=config.nside, power=-2)
                hits_map /= np.amax(hits_map)
    else:
        depth_i = [1.] * len(config.instrument.frequency)
        depth_p = [1.] * len(config.instrument.frequency)
        
    # Convert depths to requested units with CMB equivalencies
    #depth_i *= u.arcmin * u.uK_CMB
    #depth_i = depth_i.to(getattr(u, config.units) * u.arcmin, equivalencies=u.cmb_equivalencies(config.instrument.frequency * u.GHz))
    #depth_p *= u.arcmin * u.uK_CMB
    #depth_p = depth_p.to(getattr(u, config.units) * u.arcmin, equivalencies=u.cmb_equivalencies(config.instrument.frequency * u.GHz))
    bandwidths = _get_bandwidths(config, np.arange(len(config.instrument.frequency)))
    A_cmb = _get_CMB_SED(config.instrument.frequency, units=config.units, bandwidths=bandwidths)
    depth_i = depth_i * A_cmb
    depth_p = depth_p * A_cmb

    # Precompute conversion factor from arcmin to radians
    acm_to_rad = (np.pi / (180 * 60)) 

    # Get ell filter if needed
    fell = _get_ell_filter(config.lmin, config.lmax) if config.lmin > 2 else None

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
            if hp.get_nside(depth_maps_in[0]) != config.nside:
                depth_maps = np.array(
                    [np.sqrt(hp.ud_grade(dm**2, nside_out=config.nside, power=2)) for dm in depth_maps_in])
            else:
                depth_maps = np.copy(depth_maps_in)
            del depth_maps_in
        elif hasattr(config.instrument, 'path_hits_maps'):
            if not config.instrument.path_hits_maps.endswith(".fits"):
                hits_file = config.instrument.path_hits_maps + f"_{config.instrument.channels_tags[nf]}.fits"
                hits_map = hp.read_map(hits_file, field=0, dtype=np.float64)
                if hp.get_nside(hits_map) != config.nside:
                    hits_map = hp.ud_grade(hits_map, nside_out=config.nside, power=-2)
                hits_map /= np.amax(hits_map)

        if seed is not None:
            np.random.seed(seed + (nf * 3))
        # Generate noise power spectra
#        N_ell_T = (depth_i.value[nf] * acm_to_rad) ** 2 * np.ones(config.lmax + 1)
#        N_ell_P = (depth_p.value[nf] * acm_to_rad) ** 2 * np.ones(config.lmax + 1)
        N_ell_T = (depth_i[nf] * acm_to_rad) ** 2 * np.ones(config.lmax + 1)
        N_ell_P = (depth_p[nf] * acm_to_rad) ** 2 * np.ones(config.lmax + 1)
        N_ell = np.array([N_ell_T, N_ell_P, N_ell_P, 0.*N_ell_P])

        # Add knee frequency noise if provided
        if hasattr(config.instrument, 'ell_knee') and hasattr(config.instrument, 'alpha_knee'):
            ell = np.arange(config.lmax + 1)
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

        # Generate noise alm
        alm_noise = hp.synalm(N_ell, lmax=config.lmax, new=True)

        # Apply ell filter if applicable
        if fell is not None:
            for f in range(3):
                alm_noise[f] = hp.almxfl(alm_noise[f], fell)

        # Generate noise maps or alms depending on data_type
        if config.data_type=="alms":
            if hasattr(config.instrument, 'path_depth_maps'):
                noise_map = hp.alm2map(alm_noise, config.nside, lmax=config.lmax, pol=True)  * np.array([depth_maps[0], depth_maps[1], depth_maps[1]])
                noise.append(hp.map2alm(noise_map, lmax=config.lmax, pol=True, **kwargs))
            elif hasattr(config.instrument, 'path_hits_maps'):
                noise_map = hp.alm2map(alm_noise, config.nside, lmax=config.lmax, pol=True) / np.sqrt(hits_map)
                noise_map[np.isinf(noise_map)] = 0.
                noise.append(hp.map2alm(noise_map, lmax=config.lmax, pol=True, **kwargs))
            else:
                noise.append(alm_noise)
        else:
            if hasattr(config.instrument, 'path_depth_maps'):
                noise.append(hp.alm2map(alm_noise, config.nside, lmax=config.lmax, pol=True) * np.array([depth_maps[0], depth_maps[1], depth_maps[1]]))
            elif hasattr(config.instrument, 'path_hits_maps'):
                noise_map = hp.alm2map(alm_noise, config.nside, lmax=config.lmax, pol=True) / np.sqrt(hits_map)
                noise_map[np.isinf(noise_map)] = 0.
                noise.append(noise_map)
            else:
                noise.append(hp.alm2map(alm_noise, config.nside, lmax=config.lmax, pol=True))

    if config.save_inputs:
        _save_inputs(config.noise_path, np.array(noise), nsim=nsim)

    return np.array(noise)

def _get_cmb_simulation(config: Configs, nsim: Optional[Union[int, str]] = None) -> np.ndarray:
    """
    Generate simulated CMB maps or alms for a given instrument configuration.

    Parameters
    ----------
        config: Configs
            Simulation and instrument configuration. It should have the following attributes:
            - `lmax`: Maximum multipole for the simulation.
            - `nside`: HEALPix resolution.
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
    seed = None if not config.seed_cmb else (config.seed_cmb + int(nsim) if nsim is not None else config.seed_cmb)
    
    # Generating a realization of CMB alms with the loaded Cls
    alm_cmb = _get_cmb_alms_realization(cls_cmb, config.lmax, seed = seed, new = config.cls_cmb_new_ordered)
    
    # Computing the high-pass filter if lmin > 2
    fell = _get_ell_filter(config.lmin, config.lmax) if config.lmin > 2 else None

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

        cmb.append(A_cmb[idx] * alm_cmb_i if config.data_type == "alms" else A_cmb[idx] * hp.alm2map(
            alm_cmb_i, config.nside, lmax=config.lmax, pol=True
        ))
    
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

def _get_foregrounds(
    foreground_models: List[str],
    instrument: dict, 
    nside: int,
    nside_in: int,
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
            Output HEALPix resolution.
        nside_in: int
            Input HEALPix resolution. Used to apply pixel window function (if requested)
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
            instrument, sky, nside, nside_in, lmax,
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
                instrument, sky, nside, nside_in, lmax,
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
    nside_in: int,
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
            HEALPix resolution for the output.
        nside_in: int
            Input HEALPix resolution. Used to apply pixel window function (if requested)
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

    fell = _get_ell_filter(lmin, lmax) if lmin > 2 else None

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
                nside_out=nside_in if pixel_window else None
            )
        else:
            beamfile = instrument.path_beams + f"_{instrument.channels_tags[idx]}.fits"
            alm_emission = _smooth_input_alms_(
                alm_emission,
                beam_path=beamfile,
                symmetric_beam=(instrument.beams == "file_l"),
                nside_out=nside_in if pixel_window else None
            )

        if lmin > 2:
            for f in range(3):
                alm_emission[f] = hp.almxfl(alm_emission[f], fell)
        fg_component.append(alm_emission if return_alms else hp.alm2map(alm_emission, nside_out, lmax=lmax, pol=True))
        
    return np.array(fg_component)

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


