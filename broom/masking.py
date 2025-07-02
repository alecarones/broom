import numpy as np
import healpy as hp
import fnmatch
from types import SimpleNamespace
from .spectra import nmt
import os
import sys

from .routines import _slice_outputs, obj_out_to_array, _slice_data
from .configurations import Configs

REMOTE = 'https://irsa.ipac.caltech.edu/data/Planck/release_2/'
import os.path as op
from astropy.utils.data import download_file
import astropy
from typing import Any, Dict, Optional, Union, Tuple


def _preprocess_mask(mask: np.ndarray, nside_out: int) -> np.ndarray:
    """
    Preprocess a HEALPix mask by adjusting its resolution and assessing if it is binary or hits map.

    Parameters
    ----------
        mask : np.ndarray
            The input HEALPix mask, assumed to be a 1D array.
        nside_out : int
            Desired output NSIDE resolution.

    Returns
    -------
        np.ndarray
            The resampled HEALPix mask.

    Raises
    ------
        ValueError
            If the input is not a valid HEALPix mask.
    """
    if isinstance(mask, np.ndarray):
        try:
            nside_mask = hp.get_nside(mask)
#            if not is_binary_mask(mask):
#                print("Provided mask is not binary. Mask is assumed to be a hits map.")
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

def _upgrade_mask(mask: np.ndarray, nside_out: int) -> np.ndarray:
    """
    Upgrade a mask to a higher HEALPix resolution.

    Parameters
    ----------
        mask : np.ndarray
            The input mask.
        nside_out : int
            The desired NSIDE.

    Returns
    -------
        np.ndarray
            The upgraded mask.
    """

    return hp.ud_grade(mask, nside_out, power=None if is_binary_mask(mask) else -2)


def _downgrade_mask(mask: np.ndarray, nside_out: int, threshold: float = 0.5) -> np.ndarray:
    """
    Downgrade a mask to a lower HEALPix resolution.

    Parameters
    ----------
        mask : np.ndarray
            The input mask.
        nside_out : int
            The desired NSIDE.
        threshold : float, optional
            Threshold for binary conversion after downgrading.

    Returns
    -------
        np.ndarray
            The downgraded (possibly binarized) mask.
    """
    if is_binary_mask(mask):
        mask = hp.ud_grade(mask, nside_out)
        return (mask > threshold).astype(float)
    else:
        return hp.ud_grade(mask, nside_out, power=-2)

def is_binary_mask(mask: np.ndarray) -> bool:
    """
    Check if a mask is binary (contains only 0.0 and 1.0).

    Parameters
    ----------
        mask : np.ndarray
            The input mask.

    Returns
    -------
        bool
            True if binary, False otherwise.
    """
    return np.all(np.isin(mask, [0., 1.]))

def _get_mask(config: Configs, compute_cls: Dict[str, Any], nsim: Optional[str] = None) -> np.ndarray:
    """
    Determine and return the appropriate mask based on configuration and compute_cls settings.

    Parameters
    ----------
        config: Configs
            Global configuration object. It specifies:
                - mask_covariance: Path to the mask fits file which has been used in component separation. Optional. 
                            It is used only if compute_cls["mask_type"] is None.
                - nside: HEALPix resolution of the component separation products.
                - fwhm_out: Full width at half maximum of output maps.
                - lmax: Maximum multipole for the outputs and power spectra computation.   
        compute_cls: Dict[str, Any])
            Dictionary with masking and output details. It should contain:
                - mask_type: String defining the type of mask to be used (e.g., 'from_fits', 'GAL*+fgres', etc.).
                - mask_path: String defining the path to the mask file if mask_type is 'from_fits'. 
                - outputs: Object containing the outputs of the component separation.
                - path: Path to the directory where the outputs are stored.
                - field_out: String defining the field associated with the output maps (e.g., 'TQU', 'QU', etc.) loaded from path.
                - field_cls_in: String defining the fields associated with the input maps (e.g., 'TQU', 'QU', etc.) used to compute output spectra.
                                It can be a slicing of field_out.
                - fsky (Optional[float]): Final sky fraction of the mask, used if 'fgres' is in mask_type.
                - smooth_tracer (Optional[float]): Smoothing scale for the tracer used in thresholding the mask. Used if 'fgres' is in mask_type.

        nsim: str, optional
            Simulation index, used to load simulation-specific data.

    Returns
    -------
        np.ndarray
            Final mask array, potentially multi-field (shape: [n_fields, npix]).
    """
    # Derive the number of fields and pixels from the outputs
    n_fields_in = obj_out_to_array(compute_cls["outputs"]).shape[-2] if obj_out_to_array(compute_cls["outputs"]).ndim == 3 else 1
    npix = obj_out_to_array(compute_cls["outputs"]).shape[-1]
    
    # Define the mask patterns to match against compute_cls["mask_type"]
    mask_patterns = ['GAL*+fgres', 'GAL*+fgtemp', 'GAL*+fgtemp^3','GAL*0', 'GAL97', 'GAL99', 'fgres', 'fgtemp', 
            'fgtemp^3', 'config+fgres', 'config+fgtemp', 'config+fgtemp^3', 'config']

    # Case 1: No mask defined
#    if compute_cls["mask_type"] is None and config.mask_path is None:
    if compute_cls["mask_type"] is None and config.mask_observations is None and config.mask_covariance is None:
        return np.ones(npix) if n_fields_in == 1 else np.ones((n_fields_in, npix))

    # Case 2: Use config-defined mask
#    if compute_cls["mask_type"] is None and config.mask_path is not None:
#        mask_spectra = _preprocess_mask(hp.read_map(config.mask_path, field=0), config.nside)
    if compute_cls["mask_type"] is None and (config.mask_observations is not None or config.mask_covariance is not None):
        _, mask_spectra = get_masks_for_compsep(config.mask_observations, config.mask_covariance, config.nside)
        return mask_spectra if n_fields_in == 1 else np.repeat(mask_spectra[np.newaxis, :], n_fields_in, axis=0)
        
    # Case 3: Load from FITS file directly        
    if compute_cls["mask_type"] == "from_fits":
        if "mask_path" not in compute_cls or not isinstance(compute_cls["mask_path"], str):
            raise ValueError("mask_path must be a string and defined in compute_cls when mask_type is 'from_fits'")

        if n_fields_in == 1:
            return _preprocess_mask(hp.read_map(compute_cls["mask_path"], field=0), config.nside)
        
        mask_spectra = np.zeros((n_fields_in, npix))
        mask_spectra[0] = _preprocess_mask(hp.read_map(compute_cls["mask_path"], field=0), config.nside)
        if compute_cls["field_cls_in"] == "TQU":
            try:
                mask_spectra[1] = _preprocess_mask(hp.read_map(compute_cls["mask_path"], field=1), config.nside)
                mask_spectra[2] = _preprocess_mask(hp.read_map(compute_cls["mask_path"], field=1), config.nside)
            except IndexError:
                mask_spectra[1:3] = mask_spectra[0]
        elif compute_cls["field_cls_in"] in ["QU","QU_E","QU_B"]:
            mask_spectra[1] = mask_spectra[0]
        else:
            for i in range(1,n_fields_in):
                try:
                    mask_spectra[i] = _preprocess_mask(hp.read_map(compute_cls["mask_path"], field=i), config.nside)
                except IndexError:
                    mask_spectra[i] = mask_spectra[0]
        return mask_spectra

    # Case 4: Named pattern (e.g., GAL99+fgres, config, etc.)
    if any(fnmatch.fnmatch(compute_cls["mask_type"], pattern) for pattern in mask_patterns):
        # Getting the initial mask
        if 'GAL' in compute_cls["mask_type"]:
            gal_masks_list = ['GAL20','GAL40','GAL60','GAL70','GAL80','GAL90','GAL97','GAL99']
            if compute_cls["mask_type"][:5] not in gal_masks_list:
                raise ValueError("GAL mask must be one of {}".format(", ".join(gal_masks_list)))
            idx_m = gal_masks_list.index(compute_cls["mask_type"][:5])
            rot = hp.Rotator(coord=f"G{config.coordinates}") if config.coordinates != "G" else None
            mask_init = hp.ud_grade(get_planck_mask(0, field=idx_m, nside=512),hp.npix2nside(npix))
            # Applying coordinate rotation if needed
            if config.coordinates != "G":
                alm_mask = hp.map2alm(mask_init, lmax=2*hp.npix2nside(npix), pol=False)
                rot.rotate_alm(alm_mask, inplace=True)
                mask_init = hp.alm2map(alm_mask, nside_out=hp.npix2nside(npix), lmax=2*hp.npix2nside(npix), pol=False)
            mask_init = mask_init == 1.
        elif 'config' in compute_cls["mask_type"]:
#            mask_init = np.ones(npix) if config.mask_path is None else _preprocess_mask(hp.read_map(config.mask_path, field=0), config.nside)
            if config.mask_observations is not None or config.mask_covariance is not None:
                _, mask_init = get_masks_for_compsep(config.mask_observations, config.mask_covariance, config.nside)
            else:
                mask_init = np.ones(npix)
        else:
            mask_init = np.ones(npix)

        # Generating and returning the final mask according to the mask_type
        if 'fgres' in compute_cls["mask_type"] or 'fgtemp' in compute_cls["mask_type"]:
            mask_init = mask_init == 1.

            if not "fsky" in compute_cls:
                raise ValueError("fsky must be defined in compute_cls when using 'fgres' or 'fgtemp' in mask_type")
            
            if 'fgres' in compute_cls["mask_type"]:
                if not hasattr(compute_cls["outputs"], 'fgds_residuals'):
                    from .compsep import _load_outputs_
                    fgres = SimpleNamespace()
                    filename = os.path.join(
                        compute_cls["path"],
                        f"fgds_residuals/{compute_cls['field_out']}_fgds_residuals_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
                    )
                    fgres.total = _load_outputs_(filename, compute_cls["field_out"], nsim=nsim)
                    if len(compute_cls['field_out']) > 1:
                        #fgres = _slice_data(fgres, compute_cls["field_out"], compute_cls["field_cls_in"])
                        fgres = _slice_outputs(fgres,compute_cls["field_out"],compute_cls["field_cls_in"])
                    return get_threshold_mask(fgres.total,mask_init,compute_cls["field_cls_in"],compute_cls["fsky"],config.lmax,smooth_tracer=compute_cls["smooth_tracer"])
                else:
                    return get_threshold_mask(compute_cls["outputs"].fgds_residuals, mask_init, compute_cls["field_cls_in"], compute_cls["fsky"], config.lmax, smooth_tracer=compute_cls["smooth_tracer"])
            
            else:
                if 'fgres_temp_for_masking' not in compute_cls:
                    raise ValueError("Template of fgds residuals must be defined in compute_cls as 'fgres_temp_for_masking' when using 'fgtemp' in mask_type")
                
                from .compsep import _load_outputs_
                fgres = SimpleNamespace()
                filename = os.path.join(
                    compute_cls["path"],
                    f"fgres_templates/{compute_cls['fgres_temp_for_masking']}/{compute_cls['field_out']}_fgres_templates_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
                )
                fgres.total = _load_outputs_(filename, compute_cls["field_out"], nsim=nsim)
                if len(compute_cls['field_out']) > 1:
                    fgres =  _slice_outputs(fgres,compute_cls["field_out"],compute_cls["field_cls_in"])
                if 'fgtemp^3' in compute_cls["mask_type"]:
                    return get_threshold_mask(fgres.total**3,mask_init,compute_cls["field_cls_in"],compute_cls["fsky"],config.lmax,smooth_tracer=compute_cls["smooth_tracer"])
                else:
                    return get_threshold_mask(fgres.total,mask_init,compute_cls["field_cls_in"],compute_cls["fsky"],config.lmax,smooth_tracer=compute_cls["smooth_tracer"])
                                    
        else:
            return mask_init if n_fields_in == 1 else np.repeat(mask_init[np.newaxis, :], n_fields_in, axis=0)

    raise ValueError("Mask not defined. Please check 'mask_type' and 'mask_path' in compute_cls.")

def get_threshold_mask(
    map_: np.ndarray,
    mask_init: np.ndarray,
    field_cls_in: str,
    fsky: float,
    lmax: int,
    smooth_tracer: float = 3.0
) -> np.ndarray:
    """
    Apply a threshold-based masking strategy to reduce the sky fraction.

    Parameters
    ----------
        map_: np.ndarray
            Input residual tracer map(s), shape (n_fields, npix) or (npix,) to be thresholded.
        mask_init: np.ndarray
            Initial binary mask.
        field_cls_in: str
            Fields associated to map_ (e.g., "TQU", "QU").
        fsky: float
            Desired final sky fraction.
        lmax: int
            Maximum multipole to consider.
        smooth_tracer: float
            FWHM in degrees for smoothing the tracer map.

    Returns
    ----------
        np.ndarray
            Final binary mask with the desired sky fraction.
    """
    fsky_in = np.mean(mask_init**2)

    # If the initial sky fraction is already lower than the desired one, return the initial mask
    if fsky > fsky_in:
        return mask_init if map_.ndim == 1 else np.repeat(mask_init[np.newaxis, :], map_.shape[0], axis=0)

    npix = map_.shape[-1]
    npix_mask = int((fsky_in - fsky) * npix)

    # If the input map is a scalar field, apply the thresholding and return the mask
    if map_.ndim == 1:
        return threshold_scalar_tracer(map_, mask_init, npix_mask, lmax, smooth_tracer=smooth_tracer)

    mask_spectra = np.ones((map_.shape[0], npix))
    # If input tracers include polarization spin-2 fields, corresponding mask will be generated from thresholding P tracer.
    # If input tracers are scalar fields, corresponding mask will be generated from thresholding scalar tracers.
    if field_cls_in == "TQU":
        mask_spectra[0] = threshold_scalar_tracer(map_[0],mask_init,npix_mask,lmax,smooth_tracer=smooth_tracer)
        mask_spectra_P = threshold_P_tracer(map_[1:],mask_init,npix_mask,lmax,smooth_tracer=smooth_tracer)
        mask_spectra[1:3] = mask_spectra_P
    elif field_cls_in == "QU":
        mask_spectra_P = threshold_P_tracer(map_,mask_init,npix_mask,lmax,smooth_tracer=smooth_tracer)
        mask_spectra[:2] = mask_spectra_P
    else:
        for i in range(map_.shape[0]):
            mask_spectra[i] = threshold_scalar_tracer(map_[i],mask_init,npix_mask,lmax,smooth_tracer=smooth_tracer)

    return mask_spectra

def _smooth_masks(mask: np.ndarray, apodization: str, smooth_mask: float) -> np.ndarray:
    """
    Apply smoothing/apodization to a given mask or set of masks.

    Parameters
    ----------
        mask: np.ndarray
            Input binary mask(s).
        apodization: str
            Apodization type ("gaussian", "gaussian_nmt", "C1", "C2").
        smooth_mask: float
            Smoothing scale in degrees.

    Returns
    ----------
        np.ndarray
            Smoothed mask.
    """
    if mask.ndim == 1:
        return _smooth_mask(mask, apodization, smooth_mask)

    return np.array([_smooth_mask(m, apodization, smooth_mask) for m in mask])

def _smooth_mask(mask_in: np.ndarray, apodization: str, smooth_mask: float) -> np.ndarray:
    """
    Smooth a single mask using the specified apodization technique.

    Parameters
    ----------
        mask_in: np.ndarray
            Input binary mask.
        apodization: str
            Apodization type ("gaussian", "gaussian_nmt", "C1", "C2").
        smooth_mask: float
            Smoothing scale in degrees.

    Returns
    ----------
        np.ndarray
            Smoothed mask.
    """
    if apodization == "gaussian":
        return hp.smoothing(mask_in,fwhm=np.radians(smooth_mask))
    if apodization == "gaussian_nmt":
        return nmt.mask_apodization(mask_in, smooth_mask, apotype="Smooth")
    if apodization in ["C1", "C2"]:
        return nmt.mask_apodization(mask_in, smooth_mask, apotype=apodization)
    raise ValueError('apodization must be either "gaussian", "gaussian_nmt", "C1" or "C2"')

def threshold_scalar_tracer(
    map_: np.ndarray,
    mask_in: np.ndarray,
    npix_mask: int,
    lmax: int,
    smooth_tracer: float = 3.0
) -> np.ndarray:
    """
    Generate a threshold mask from a scalar tracer map.

    Parameters
    ----------
        map_: np.ndarray
            Scalar tracer map.
        mask_in: np.ndarray
            Initial mask.
        npix_mask: int
            Number of additional pixels to mask.
        lmax: int
            Maximum multipole to be considerd for smoothing the tracer map.
        smooth_tracer: float
            Smoothing FWHM in degrees.

    Returns
    ----------
        np.ndarray
            Threshold mask.
    """

    smoothed_tracer = hp.smoothing(np.absolute(map_),fwhm=np.radians(smooth_tracer),lmax=lmax,pol=False)
    #smoothed_tracer = np.absolute(hp.smoothing(map_,fwhm=np.radians(smooth_tracer),lmax=lmax,pol=False))

    mask_spectra = np.ones_like(map_)
    idx_mask = np.argsort(smoothed_tracer * mask_in)[-npix_mask:]
    mask_spectra[idx_mask] = 0.
    mask_spectra[mask_in == 0.] = 0.
    return mask_spectra

def threshold_P_tracer(
    map_: np.ndarray,
    mask_in: np.ndarray,
    npix_mask: int,
    lmax: int,
    smooth_tracer: float = 3.0
) -> np.ndarray:
    """
    Generate a threshold mask from polarization tracers maps (Q, U).

    Parameters
    ----------
        map_: np.ndarray
            Polarization tracers maps [Q, U]. Shape: (2, npix).
        mask_in: np.ndarray
            Initial mask.
        npix_mask: int
            Number of pixels to mask.
        lmax: int
            Maximum multipole to be considerd for smoothing the tracers maps.
        smooth_tracer: float
            Smoothing FWHM in degrees.

    Returns
    ----------
        np.ndarray: 
            Thresholded polarization mask.
    """

    mask_spectra = np.ones_like(map_[0])
    map_ = hp.smoothing([0.*map_,map_[0],map_[1]],fwhm=np.radians(smooth_tracer),lmax=lmax,pol=True)[1:]
    map_P = np.sqrt((map_[0])**2 + (map_[1])**2)
    idx_mask = np.argsort(map_P * mask_in)[-npix_mask:]  #
    mask_spectra[idx_mask] = 0.
    mask_spectra[mask_in == 0.] = 0.
    return mask_spectra

def _get_processed_dir():
    """
    Returns the path to the processed Planck directory in the astropy cache,
    creating it if necessary.

    Returns
    -------
        str: 
            Directory path.
    """
    processed_dir = op.join(astropy.config.get_cache_dir(),
                            'processed', 'planck')
    if not op.exists(processed_dir):
        os.makedirs(processed_dir)

    return processed_dir

def get_planck_mask(apo: int = 5, nside: int = 2048, field: int = 3, info: bool = False) -> Union[np.ndarray, tuple]:
    """
    Downloads and processes a Planck Galactic mask.

    Parameters
    -------    
        apo: int 
            Apodization level (default 5).
        nside: int 
            HEALPix resolution.
        field: int 
            Field to read.
        info: bool 
            Whether to return header info.

    Returns
    -------
        np.ndarray or tuple: 
            Mask data or (mask, header).
    """
    remote_url = op.join(REMOTE, 'ancillary-data/masks',
                         f'HFI_Mask_GalPlane-apo{apo}_2048_R2.00.fits')
    remote_file = download_file(remote_url, cache=True)

    if info:
        return hp.read_map(remote_file, field=[], h=True)

    local_file = op.join(_get_processed_dir(),
                         f'HFI_Mask_GalPlane-apo{apo}_{nside}_R2.00.fits')
    try:
        return hp.read_map(local_file, field=field)
    except IOError:
        output = hp.read_map(remote_file, field=None)
        output = hp.ud_grade(output, nside)
        hp.write_map(local_file, output)
        return output[field]

def get_masks_for_compsep(mask_obs: Union[str, np.ndarray, None]
    , mask_cov: Union[str, np.ndarray, None]
    , nside: int) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    """
    Get observation and covariance masks for component separation.

    Parameters
    ----------
        mask_obs: Union[str, np.ndarray, None]
            Path to the observation mask fits file or a numpy array. If None, no observation mask is applied.
        mask_cov: Union[str, np.ndarray, None]
            Path to the covariance mask fits file or a numpy array. If None, the observation mask is used as the covariance mask.
        nside: int
            HEALPix resolution for the masks in output.
    
    Returns
    -------
        Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
            Tuple containing the observation mask and covariance mask as numpy arrays. If no mask is provided, returns None.
    """

    if mask_obs is not None:
        if not isinstance(mask_obs, (str, np.ndarray)):
            raise ValueError("mask_observations must be a string full path to a HEALPix mask fits file or a numpy array.")

        if isinstance(mask_obs, str):
            mask_obs = hp.read_map(mask_obs, field=0)
        mask_obs = _preprocess_mask(mask_obs, nside)
        mask_obs /= np.max(mask_obs)  # Normalizing mask to have values between 0 and 1
    else:
        mask_obs = None
        
    if mask_cov is not None:
        if not isinstance(mask_cov, (str, np.ndarray)):
            raise ValueError("mask_covariance must be a string full path to a HEALPix mask fits file or a numpy array.")

        if isinstance(mask_cov, str):
            mask_cov = hp.read_map(mask_cov, field=0)
        mask_cov = _preprocess_mask(mask_cov, nside)
        mask_cov /= np.max(mask_cov)  # Normalizing mask to have values between 0 and 1
        if mask_obs is not None:
            mask_cov[mask_obs == 0.] = 0.  # Ensuring that the covariance mask is zero where the observation mask is zero
            mask_obs[mask_cov == 0.] = 0.  # Ensuring that the observation mask is zero where the covariance mask is zero
    else:
        mask_cov = mask_obs

    return mask_obs, mask_cov


__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]
                    
