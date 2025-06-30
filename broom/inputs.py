import numpy as np
import healpy as hp
import re
import sys
from .configurations import Configs
from .leakage import purify_master, purify_recycling
from types import SimpleNamespace
from .routines import _log, _get_ell_filter, _bl_from_fwhms, _bl_from_file
from typing import Any, Optional, Union, Dict
from types import SimpleNamespace



def _alms_from_data(
    config: Configs,
    data,
    field_in: str,
    mask_in: np.ndarray = None,
    data_type: str = "maps",
    bring_to_common_resolution: bool = True,
    pixel_window_in: bool = False,
    **kwargs
) -> SimpleNamespace:
    """Return spherical harmonic coefficients (alms) from input map or alm multifrequency data.

    Parameters
    ----------
        config: Configs
            Configuration object. It should contain settings as:
                - lmax: Maximum multipole for spherical harmonics.
                - nside_in: HEALPix nside associated with input maps or alms.
                - field_out: Desired output field type from component separation (e.g., "T", "E", "B", "QU", "TEB").
                - leakage_correction: Leakage correction method (e.g., "mask_only", "purify_master_B", "purify_recycling_B").
                - fwhm_out: Desired angular resolution FWHM of output products in arcminutes.
                - instrument: Instrument configuration, which should include:
                    - beams: Type of beam (e.g., "gaussian", "file_l", "file_lm").
                    - fwhms: FWHM of the instrument beams in arcminutes. Used if beams is "gaussian".
                    - path_beams: Path to the beam files if beams is "file_l" or "file_lm".
                    - channels_tags: Tags for the instrument channels, used to identify beam files.
                - verbose: Whether to print additional information during processing.    
        data: SimpleNamespace
            Namespace containing input data. Each attribute should be a 2D or 3D NumPy array.
        field_in: str
            Field type associated to input data (e.g., "TQU", "EB").
        mask_in: np.ndarray, optional
            Mask to apply to maps. To be applied in case of simulations of partial-sky observations.
            Default: None.
        data_type: str, optional 
            Type of data ('maps' or 'alms'). Default: 'maps'.
        bring_to_common_resolution: bool, optional
            Whether to harmonize resolution of input data. Default: True.
            If False, it is assumed that all input data are already at the same resolution.    
        pixel_window_in:  bool, optional
            Whether to apply pixel window correction. Default: False.
        **kwargs: 
            Additional arguments passed to map2alm.

    Returns
    -------
        SimpleNamespace
            Alms associated to input data.
    """
    fell = _get_ell_filter(2,config.lmax)

    if data_type == "maps":
        alms = _maps_to_alms(config, data, field_in, mask_in=mask_in, **kwargs)
    elif data_type == "alms":
        alms = _alms_to_alms(config, data)

    alms = _processing_alms(config, alms, bring_to_common_resolution=bring_to_common_resolution, pixel_window_in=pixel_window_in)    

    return alms

def _maps_to_alms(
    config: Configs,
    data: SimpleNamespace,
    field_in: str,
    mask_in: Union[np.ndarray, None] = None,
    **kwargs
) -> SimpleNamespace:
    """
    Convert HEALPix maps to spherical harmonic coefficients alms.

    Parameters
    ----------
        config : Configs
            Configuration object containing parameters such as lmax, leakage correction, field_out, etc.
        data : SimpleNamespace
            Namespace containing map data arrays. Each attribute should be a 2D or 3D NumPy array shaped 
            as (n_channels, [n_fields], n_pixels).
        field_in : str
            The input field type, e.g. "T", "QU", "TEB", or "EB".
        mask_in : np.ndarray or None, optional
            Optional mask to apply before harmonic transform. If None, a mask of ones is used.
        kwargs : dict
            Additional keyword arguments passed to `hp.map2alm`.

    Returns
    -------
        alms : SimpleNamespace
            Namespace containing harmonic coefficients for each map component with matching structure to `data`.
    """
    fell = _get_ell_filter(2,config.lmax)

    # Validate and prepare mask
    if mask_in is None:
        mask_in = np.ones(data.total.shape[-1])
    elif not isinstance(mask_in, np.ndarray): 
        raise ValueError("Invalid mask. It must be a numpy array.")
    elif hp.get_nside(mask_in) != hp.npix2nside(data.total.shape[-1]):
            raise ValueError("Mask HEALPix resolution does not match data HEALPix resolution.")

    mask_in /= np.max(mask_in)
    
    alms = SimpleNamespace()

    for attr_name in vars(data):
        attr_maps = getattr(data, attr_name)

        alms_attr = _attr_maps_to_alms(config, attr_maps, field_in, mask_in, fell,
            total_maps=data.total if config.leakage_correction and "_recycling" in config.leakage_correction else None, **kwargs)

        setattr(alms, attr_name, alms_attr)

    return alms


def _attr_maps_to_alms(config: Configs, maps, field_in, mask_in, fell, total_maps=None, **kwargs):
    """
    Convert a specific SimpleNamespace attribute composed of maps into alms.

    Parameters
    ----------
        config : Configs
            Configuration object containing parameters such as lmax, leakage correction, field_out, etc. See _alms_from_data for details.
        maps : np.ndarray
            Maps to be converted to alms. Should be shaped as (n_channels, [n_fields], n_pixels).
        field_in : str
            The input field type associated to provided maps, e.g. "T", "QU", "TEB", or "EB".
        mask_in : np.ndarray
            Optional mask to apply before harmonic transform. 
            If None, a mask of ones is used.
        fell : np.ndarray
            Filter to apply to the alms.
        total_maps : np.ndarray, optional
            Total maps used for leakage correction, if applicable. Should be shaped as (n_channels, [n_fields], n_pixels).
        kwargs : dict
            Additional keyword arguments passed to `hp.map2alm`.
    
    Returns
    -------
        alms_arr : np.ndarray
            Alms array with shape (n_channels, [n_fields], n_alms) where n_alms is hp.Alm.getsize(config.lmax).
    """  

    n_channels = maps.shape[0]

    # Single scalar field case
    if maps.ndim == 2:
        alms_arr = np.zeros((n_channels, hp.Alm.getsize(config.lmax)), dtype=complex)
    else:
        # Alms from TQU or TEB multifrequency maps
        if maps.shape[1]==3:
            alms_arr = np.zeros((n_channels, 3, hp.Alm.getsize(config.lmax)), dtype=complex)
        # Alms from EB or QU multifrequency maps
        elif maps.shape[1]==2:
            is_single_field = config.field_out in ["E", "QU_E", "B", "QU_B"]
            shape = (n_channels, hp.Alm.getsize(config.lmax)) if is_single_field else (n_channels, 2, hp.Alm.getsize(config.lmax))
            alms_arr = np.zeros(shape, dtype=complex)

    for i in range(n_channels):
        alms_arr[i] = _maps_to_alms_channel(config, maps[i], field_in, mask_in, fell, total_maps=total_maps[i] if total_maps is not None else None, **kwargs)

    return alms_arr
                
def _maps_to_alms_channel(config: Configs, maps, field_in, mask_in, fell, total_maps=None, **kwargs):
    """
    Convert a single frequency channel set of maps into alms.

    Parameters
    ----------
        config : Configs
            Configuration object containing parameters such as lmax, leakage correction, field_out, etc. See _alms_from_data for details.
        maps : np.ndarray
            Maps to be converted to alms. Should be shaped as ([n_fields,] n_pixels).
        field_in : str
            The input field type associated to provided maps, e.g. "T", "QU", "TEB", or "EB".
        mask_in : np.ndarray
            Optional mask to apply before harmonic transform. If None, a mask of ones is used.
        fell : np.ndarray
            Filter to apply to the alms.
        total_maps : np.ndarray, optional
            Total maps used for leakage correction, if applicable. Should be shaped as ([n_fields,] n_pixels).
        kwargs : dict
            Additional keyword arguments passed to `hp.map2alm`.
    
    Returns
    -------
        alms_out : np.ndarray
            Alms array with shape ([n_fields,] n_alms) where n_alms is hp.Alm.getsize(config.lmax).
    """
    
    mask_bin_in = np.ceil(mask_in)

    # Extract leakage correction iterations
    if config.leakage_correction is not None:
        if "_recycling" in config.leakage_correction:
            if "_iterations" in config.leakage_correction:
                iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
            else:
                iterations = 0

    if maps.ndim == 1:
        return hp.almxfl(hp.map2alm(maps * mask_bin_in, lmax=config.lmax, pol=False, **kwargs), fell)
    
    else:
        if maps.shape[0] == 3:
            if (config.leakage_correction is None) or (field_in=="TEB") or (np.mean(mask_in**2)==1.):
                alms_out = hp.map2alm(maps * mask_bin_in, lmax=config.lmax, pol=field_in=="TQU", **kwargs)
            else:
                alms_out = np.zeros((3, hp.Alm.getsize(config.lmax)), dtype=complex)
                alms_out[0] = hp.map2alm((maps[0] * mask_bin_in), lmax=config.lmax, **kwargs)
#                if config.leakage_correction == "mask_only":
#                    alms_out[1:] = hp.map2alm(maps * mask_in, lmax=config.lmax, pol=True, **kwargs)[1:]
#                elif "_purify" in config.leakage_correction:
                if "_purify" in config.leakage_correction:
                    alms_out[1:] = purify_master(maps[1:], mask_in, config.lmax,purify_E=("E" in config.leakage_correction))
                elif "_recycling" in config.leakage_correction:
                    alms_out[1:] = purify_recycling(maps[1:], total_maps[1:], mask_bin_in, config.lmax, purify_E=("E" in config.leakage_correction), iterations=iterations, **kwargs)
            
            for j in range(3):
                alms_out[j] = hp.almxfl(alms_out[j], fell)
            return alms_out

        elif maps.shape[0] == 2:
            is_single_field = config.field_out in ["E", "QU_E", "B", "QU_B"]

            if (
            config.leakage_correction is None or
            np.mean(mask_in ** 2) == 1. or
            field_in == "EB" or
            (config.field_out in ["E", "QU_E"] and "E" not in config.leakage_correction) or
            (config.field_out in ["B", "QU_B"] and "B" not in config.leakage_correction)
            ):
                alms_out = hp.map2alm(np.vstack((0. * maps[0], maps)) * mask_bin_in, lmax=config.lmax, pol=field_in=="QU", **kwargs)[1:]
            else:
#                if config.leakage_correction == "mask_only":
#                    alms_out = hp.map2alm(np.vstack((0. * maps[0], maps)) * mask_in, lmax=config.lmax, pol=True, **kwargs)[1:]
#                elif "_purify" in config.leakage_correction:
                if "_purify" in config.leakage_correction:
                    alms_out = purify_master(maps, mask_in, config.lmax, purify_E=("E" in config.leakage_correction))
                elif "_recycling" in config.leakage_correction:
                    alms_out = purify_recycling(
                    maps, total_maps, mask_bin_in, config.lmax,
                    purify_E=("E" in config.leakage_correction),
                    iterations=iterations,
                    **kwargs
                    )

            if is_single_field:
                idx = 0 if config.field_out in ["E", "QU_E"] else 1
                return hp.almxfl(alms_out[idx], fell)
            else:
                for j in range(2):
                    alms_out[j] = hp.almxfl(alms_out[j], fell)
            return alms_out
        

def _alms_to_alms(
    config: Configs,
    data: SimpleNamespace,
) -> SimpleNamespace:
    """
    Process provided input alms into coefficients arrays ready to be used in the component separation pipeline.

    Parameters
    ----------
        config : Configs
            Configuration object with attributes such as field_out and lmax.
        data : SimpleNamespace
            Namespace containing input alms. Each attribute should be shaped as:
            - (n_channels, n_alms) for scalar inputs,
            - (n_channels, 2, n_alms) for EB,
            - (n_channels, 3, n_alms) for TEB.

    Returns
    -------
        alms : SimpleNamespace
            Namespace with transformed alm coefficients matching `config.field_out` and to be used in the component separation pipeline.
    """
    fell = _get_ell_filter(2,config.lmax)
    alms = SimpleNamespace()

    for attr_name in vars(data):
        alms_input = getattr(data, attr_name)
        n_channels = alms_input.shape[0]
        lmax_in = hp.Alm.getlmax(alms_input.shape[-1])

        if lmax_in == config.lmax:
            alms_attr = np.copy(alms_input)
        else:
            target_shape = (n_channels, hp.Alm.getsize(config.lmax)) if alms_input.ndim == 2 else (n_channels, alms_input.shape[1], hp.Alm.getsize(config.lmax))
            alms_attr = np.zeros(target_shape, dtype=complex)
            
            idx_lmax = np.array([hp.Alm.getidx(lmax_in, ell, m) for ell in range(2, config.lmax + 1) for m in range(ell + 1)])
            idx_config_lmax = np.array([hp.Alm.getidx(config.lmax, ell, m) for ell in range(2, config.lmax + 1) for m in range(ell + 1)])

            alms_attr[..., idx_config_lmax] = alms_input[..., idx_lmax]

        del alms_input

        if alms_attr.ndim == 2:
            for i in range(alms_attr.shape[0]):
                alms_attr[i] = hp.almxfl(alms_attr[i], fell)
        elif alms_attr.ndim == 3:
            for i, j in np.ndindex(alms_attr.shape[0], alms_attr.shape[1]):
                alms_attr[i, j] = hp.almxfl(alms_attr[i, j], fell)

        if alms_attr.ndim == 3 and alms_attr.shape[1]==2 and config.field_out in ["E", "QU_E"]:
            setattr(alms, attr_name, alms_attr[:, 0, :])
        elif alms_attr.ndim == 3 and alms_attr.shape[1]==2 and config.field_out in ["B", "QU_B"]:
            setattr(alms, attr_name, alms_attr[:, 1, :])
        else:
            setattr(alms, attr_name, alms_attr)

    return alms

def _processing_alms(
    config: Configs,
    alms: SimpleNamespace,
    bring_to_common_resolution: bool = True,
    pixel_window_in: bool = False
) -> SimpleNamespace:
    """
    Preprocess Alm coefficients including:
    - Bringing all inputs to a common beam resolution.
    - Correcting for the input pixel window function.

    Parameters
    ----------
        config : Configs
            Configuration object containing field_out, instrument settings, etc.
        alms : SimpleNamespace
            Input alm coefficients with shape (n_channels, (n_fields), n_alms).
        bring_to_common_resolution : bool, optional
            Whether to bring all inputs to a common angular resolution. Default is True.
        pixel_window_in : bool, optional
            Whether to correct input for the HEALPix pixel window function. Default is False.

    Returns
    -------
        SimpleNamespace
            Processed Alm coefficients.
    """
    if bring_to_common_resolution:
        _log("Bringing inputs to common resolution", verbose=config.verbose)
        alms = _bring_to_common_resolution(config, alms)
    else:
        _log("Inputs are assumed to be at common angular resolution", verbose=config.verbose)
    if pixel_window_in:
        _log("Correcting for input pixel window function", verbose=config.verbose)
        alms = _correct_input_pixwin(config, alms)
    return alms

def _bring_to_common_resolution(
    config: Configs,
    alms: SimpleNamespace
    ) -> SimpleNamespace:
    """
    Brings each Alm channel to the target resolution.

    Parameters
    ----------
        config : Configs
            Configuration object including instrument, lmax, and output resolution (fwhm_out).
        alms : SimpleNamespace
            Namespace with Alm arrays of shape (n_channels, (n_fields), n_alms).

    Returns
    -------
        SimpleNamespace
            Alm coefficients adjusted to the common beam resolution.
    """

    if alms.total.ndim == 2:
        idx_bl = {
            "T": 0,
            "E": 1, "QU_E": 1,
            "B": 2, "QU_B": 2
        }.get(config.field_out)
        if idx_bl is None:
            raise ValueError(f"Invalid field_out for 2D alms: {config.field_out}")
    elif alms.total.ndim == 3:
        idx_bl = [0,1,2] if alms.total.shape[1] == 3 else [1,2]
    
    if config.instrument.beams not in ["gaussian", "file_l", "file_lm"]:
        raise ValueError("instrument.beams must be 'gaussian', 'file_l', or 'file_lm'")

    for i in range(alms.total.shape[0]):  # Loop over channels
        if config.instrument.beams == "gaussian":
            bl = _bl_from_fwhms(config.fwhm_out,config.instrument.fwhm[i],config.lmax)
        else:
            bl = _bl_from_file(
                config.instrument.path_beams,
                config.instrument.channels_tags[i],
                config.fwhm_out,
                config.instrument.beams,
                config.lmax
            )

        for attr_name in vars(alms):
            alm = getattr(alms, attr_name)[i]

            if alm.ndim == 1:
                if config.instrument.beams == "file_lm":
                    alm = alm * bl[:,idx_bl]
                else:
                    alm = hp.almxfl(alm, bl[:,idx_bl])
            elif alm.ndim == 2:
                for j in range(alm.shape[0]):
                    if config.instrument.beams == "file_lm":
                        alm[j] = alm[j] * bl[:, idx_bl[j]]
                    else:
                        alm[j] = hp.almxfl(alm[j], bl[:, idx_bl[j]])

            getattr(alms, attr_name)[i] = alm
            del alm

    return alms

def _correct_input_pixwin(
    config: Configs,
    alms: SimpleNamespace
) -> SimpleNamespace:
    """
    Applies correction for the HEALPix input pixel window function.

    Parameters
    ----------
        config : Configs
            Configuration with input nside and lmax.
        alms : SimpleNamespace
            Namespace with Alm arrays to be corrected.

    Returns
    -------
        SimpleNamespace
            Pixel-window corrected Alm coefficients.
    """

    pixwin_inv = 1. / np.array(hp.pixwin(config.nside_in, pol=True, lmax=config.lmax))
    pixwin_inv[np.isinf(pixwin_inv)] = 0.

    for attr_name in vars(alms):
        alm = getattr(alms, attr_name)

        if alm.ndim == 2:
            pw = pixwin_inv[0] if config.field_out == "T" else pixwin_inv[1]
            for i in range(alm.shape[0]):
                alm[i] = hp.almxfl(alm[i], pw)
        elif alm.ndim == 3:
            n_fields = alm.shape[1]
            for j in range(n_fields):
                pw = pixwin_inv[0] if (j == 0 and n_fields == 3) else pixwin_inv[1]
                for i in range(alm.shape[0]):
                    alm[i, j] = hp.almxfl(alm[i, j], pw)

        setattr(alms, attr_name, alm)
        del alm

    return alms

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]
                    

