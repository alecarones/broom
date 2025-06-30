import numpy as np
import healpy as hp
import mtneedlet as nl
from typing import Dict, Union, List, Tuple, Optional
from .configurations import Configs
from .leakage import purify_master, purify_recycling
from types import SimpleNamespace
import warnings
import sys


def _get_needlet_windows_(needlet_config: Dict, lmax: int) -> np.ndarray:
    """
    Dispatch function to generate needlet windows based on configuration.

    Parameters
    ----------
        needlet_config : dict
            Dictionary containing needlet settings:
            - "needlet_windows": Type of needlet windows ('cosine', 'standard', 'mexican').
            - "ell_peaks": List of integers defining multipoles of peaks for needlet bands (required for 'cosine').
            - "width": Width of the needlet windows (required for 'standard' and 'mexican').
            - "merging_needlets": Integer or list of integers defining ranges of needlets to be merged.
        lmax : int
            Maximum multipole.

    Returns
    -------
        np.ndarray
            Needlet window array of shape (n_bands, lmax+1)
    """
    needlet_options = {
    "cosine": get_cosine_windows,
    'standard': get_standard_windows,
    'mexican': get_mexican_windows}

    if needlet_config["needlet_windows"] not in needlet_options:
        raise ValueError(f"Unsupported needlet type: {needlet_config['needlet_windows']}")

    return needlet_options[needlet_config["needlet_windows"]](lmax, needlet_config)
    
def get_cosine_windows(lmax: int, needlet_config: Dict) -> np.ndarray:
    """
    Generate cosine-based needlet windows.

    Parameters
    ----------
        lmax : int
            Maximum multipole.
        needlet_config : dict
            Configuration with key "ell_peaks" (list of peak positions).

    Returns
    -------
        np.ndarray
            Cosine window array (n_bands, lmax+1).
    """
    if "ell_peaks" not in needlet_config:
        raise ValueError("Cosine needlets require 'ell_peaks' in the config. \
            'ell_peaks' is a list of integers that define the multipoles of the peaks of the cosine windows.")

    def compute_cosine_window(ell_peak: int, ell_ext: int) -> np.ndarray:
        ell_range = np.arange(min(ell_peak, ell_ext), max(ell_peak, ell_ext) + 1)
        return np.cos((ell_peak - ell_range) / (ell_peak - ell_ext) * np.pi / 2.)

    bandpeaks = needlet_config["ell_peaks"]
    n_bands = len(bandpeaks)
    if n_bands < 2:
        warnings.warn("Just one peak provided, no needlets are generated and the method will be performed in pixel domain.",UserWarning)
        return np.ones((1, lmax + 1))
    b_ell = np.zeros((n_bands,lmax+1))

    bandpeaks = [peak for peak in bandpeaks if peak <= lmax]
    bandpeaks = sorted(bandpeaks)

    if bandpeaks[0] > 0:
        b_ell[0,:bandpeaks[0]]=1.

    b_ell[0,bandpeaks[0]:bandpeaks[1]+1] = compute_cosine_window(bandpeaks[0], bandpeaks[1]) #

    if n_bands >= 3:
        for i in range(1, n_bands-1):
            ell_min, ell_p, ell_max = bandpeaks[i - 1], bandpeaks[i], bandpeaks[i + 1]
            b_ell[i,ell_min:ell_p+1] = compute_cosine_window(ell_p, ell_min)
            b_ell[i,ell_p:ell_max+1] = compute_cosine_window(ell_p, ell_max)

    b_ell[-1,bandpeaks[-2]:bandpeaks[-1]+1] = compute_cosine_window(bandpeaks[-1], bandpeaks[-2])
    if bandpeaks[-1] < lmax:
        b_ell[-1,bandpeaks[-1]+1:] = 1.
    return b_ell

def get_standard_windows(lmax: int, needlet_config: Dict) -> np.ndarray:
    """
    Generate standard needlet windows.

    Parameters
    ----------
        lmax : int
            Maximum multipole.
        needlet_config : dict
            Must contain 'width' and optionally 'merging_needlets'.

    Returns
    -------
        np.ndarray
            Standard needlet windows.
    """
    if not "width" in needlet_config:
        raise ValueError("'standard' needlet windows require 'width' parameter. \
            'width' is a floating number that defines the width of the standard needlet windows.")

    j_min, j_max = 0, 2

    while True:
        j_range = list(range(j_min, j_max + 1))
        b_ell = nl.standardneedlet(needlet_config["width"], j_range, lmax)
        if np.abs(np.sum(b_ell**2, axis=0)[-1] - 1.0) <= 1e-5:
            break
        j_max += 1

    return _merge_needlets_(b_ell, needlet_config.get("merging_needlets"))

def get_mexican_windows(lmax: int, needlet_config: Dict) -> np.ndarray:
    """
    Generate Mexican needlet windows.

    Parameters
    ----------
        lmax : int
            Maximum multipole.
        needlet_config : dict
            Must contain 'width' and optionally 'merging_needlets'.

    Returns
    -------
        np.ndarray
            Mexican needlet windows.
    """

    if "width" not in needlet_config:
        raise ValueError("'mexican' needlet windows require 'width' parameter. \
            'width' is a floating number that defines the width of the mexican needlet windows.")

    j_min, j_max = 0, 2
    width = needlet_config["width"]

    while True:
        j_range = list(range(j_min, j_max + 1))
        b_ell = nl.mexicanneedlet(needlet_config["width"], j_range, lmax)
        if np.abs(np.sum(b_ell**2, axis=0)[-1] - 1.0) <= 1e-5:
            break
        j_max += 1
    
    return _merge_needlets_(b_ell, needlet_config.get("merging_needlets"))

    
def _merge_needlets_(b_ell: np.ndarray, merging_needlets: Optional[Union[int, List[int]]]) -> np.ndarray:
    """
    Merge needlet bands according to merging strategy.

    Parameters
    ----------
        b_ell : np.ndarray
            Original needlet windows with shape (n_bands, lmax+1).
        merging_needlets : int or list of int or None
            Strategy for merging bands. If None no merging is performed.
            If int, merges first `merging_needlets` bands into one.
            If list, merges bands according to the ranges defined in the list.

    Returns
    -------
        np.ndarray
            Merged needlet windows.
    """
    if merging_needlets is None:
        return b_ell

    n_bands, lmaxp1 = b_ell.shape

    if isinstance(merging_needlets, int):
        if n_bands <= merging_needlets:
            warnings.warn("Number of needlets to merge in the first band is larger than the number of needlets. \
                All needlets are merged in one band. You are therefore running in pixel domain.",UserWarning)
            return np.sqrt(np.sum(b_ell**2, axis=0))[None, :]
        merged_bell = np.sqrt(np.sum((b_ell**2)[:merging_needlets],axis=0))[None, :]
        return np.concatenate([merged_bell, b_ell[merging_needlets:]])

    elif isinstance(merging_needlets, list):
        merging_needlets = sorted(set(merging_needlets))

        if b_ell.shape[0] <= merging_needlets[0]:
            warnings.warn("Number of needlets to merge in the first band is larger than the number of needlets. \
                All needlets are therefore merged in one band and the method will be performed in pixel domain.",UserWarning)
            return  np.sqrt(np.sum(b_ell**2, axis=0))[None, :]

        if merging_needlets[0] > 0:
            merging_needlets = [0] + merging_needlets
        if merging_needlets[-1] >= n_bands:
            merging_needlets = [x for x in merging_needlets if x < n_bands]
            merging_needlets.append(n_bands)

        merged_b_ell = []
        for start, end in zip(merging_needlets[:-1], merging_needlets[1:]):
            merged_b_ell.append(np.sqrt(np.sum((b_ell**2)[start:end], axis=0)))
        merged_b_ell = np.array(merged_b_ell)
        if merging_needlets[-1] < n_bands:
            merged_b_ell = np.concatenate([merged_b_ell, b_ell[merging_needlets[-1]:]])
        return merged_b_ell
    else:
        raise ValueError("merging_needlets must be int, list[int], or None")

def _get_nside_lmax_from_b_ell(b_ell: np.ndarray, nside: int, lmax: int) -> Tuple[int, int]:
    """
    Estimate optimal nside and lmax based on the needlet filter support.

    Parameters
    ----------
        b_ell : np.ndarray
            Needlet filters of shape (n_bands, lmax+1).
        nside : int
            Input nside.
        lmax : int
            Input lmax.

    Returns
    -------
        Tuple[int, int]
            Recommended (nside, lmax) values to be used for needlet maps generation.
    """

    max_b = np.max(np.nonzero(b_ell)) 
    if max_b == lmax:
        return nside, lmax
    
    max_b = max_b / 2

    # If the band is non zero only for ell <= 16, nside and lmax are set to 8 and 16
    if max_b <= 8:
        return 8, 16
    
    for k in range(3, 12):
        lower, upper = 2**k, 2**(k + 1)
        if lower < max_b <= upper:
            if upper <= nside:
                return upper, 2 * upper
            else:
                return nside, 2 * nside
    return nside, lmax

def _needlet_filtering(alms: np.ndarray, b_ell: np.ndarray, lmax_out: int) -> np.ndarray:
    """
    Filter alm coefficients using a given needlet filter.

    Parameters
    ----------
        alms : np.ndarray
            Input spherical harmonic coefficients of shape (n_alms, n_components) or (n_alms,).
        b_ell : np.ndarray
            Needlet filter of length lmax+1.
        lmax_out : int
            Desired output lmax.

    Returns
    -------
        np.ndarray
            Filtered and possibly resized alm coefficients.
    """
    if alms.ndim == 2:
        filtered_alms = np.array([hp.almxfl(alms[:,c], b_ell) for c in range(alms.shape[-1])]).T
    elif alms.ndim == 1:
        filtered_alms = hp.almxfl(alms, b_ell)

    if alms.shape[0] == hp.Alm.getsize(lmax_out):
        return filtered_alms
    else:
        lmax_j = min(hp.Alm.getlmax(alms.shape[0]), lmax_out)
        idx_lmax_in = np.array([hp.Alm.getidx(hp.Alm.getlmax(filtered_alms.shape[0]), ell, m) for ell in range(0, lmax_j+1) for m in range(ell + 1)])
        idx_lmax_out = np.array([hp.Alm.getidx(lmax_out, ell, m) for ell in range(0, lmax_j+1) for m in range(ell + 1)])
        if alms.ndim == 2:
            alms_j = np.zeros((hp.Alm.getsize(lmax_out), alms.shape[-1]), dtype=complex)
            alms_j[idx_lmax_out, :] = filtered_alms[idx_lmax_in, :]
        elif alms.ndim == 1:
            alms_j = np.zeros((hp.Alm.getsize(lmax_out)), dtype=complex)
            alms_j[idx_lmax_out] = filtered_alms[idx_lmax_in]
        return alms_j

def _get_good_channels_nl(config: Configs, b_ell: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """
    Returns indices of frequency channels with acceptable beam transfer function ratios.

    Parameters
    ----------
        config : Configs
            Configuration object with instrument parameters.
        b_ell : np.ndarray
            Needlet harmonic window.
        threshold : float
            Minimum acceptable ratio of input/output beam values.

    Returns
    -------
        np.ndarray
            Indices of "good" channels.
    """
    freqs_nl = []
    for i, fwhm in enumerate(config.instrument.fwhm):
        bl_in = hp.gauss_beam(np.radians(fwhm/60.), lmax=config.lmax,pol=False)
        bl_out = hp.gauss_beam(np.radians(config.fwhm_out/60.), lmax=config.lmax,pol=False)
        if not np.any((bl_in / bl_out)[b_ell > 1e-2] < threshold):
            freqs_nl.append(i)
    return np.array(freqs_nl)

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]
                    
