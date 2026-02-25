import numpy as np
import healpy as hp
from astropy.io import fits
from .configurations import Configs
from types import SimpleNamespace
from typing import Optional, Union, List, Dict, Any
import sys

def _get_ell_filter(lmin: int, lmax: int) -> np.ndarray:
    """
    Return a filter array in harmonic space with filtering at low multipoles.
    
    Parameters
    ----------
        lmin : int
            Minimum multipole to consider.
        lmax : int
            Maximum multipole to consider.
    
    Returns
    -------
        fell : np.ndarray
            Filter array of shape (lmax+1,) with values set to 1 for multipoles >= lmin,
            and 0 for multipoles < lmin, with a cosine transition for low multipoles.
    """
    fell = np.ones(lmax+1)
    if lmin <=3:
        fell[:lmin] = 0.
    elif lmin <= 6:
        fell[:2]=0.
        fell[2:lmin+1]=np.cos((lmin - np.arange(2,lmin+1)) * np.pi / (2. * (lmin-2)))
    else:
        fell[:lmin-4]=0.
        fell[lmin-4:lmin+1]=np.cos((lmin - np.arange(lmin-4,lmin+1)) * np.pi / 8.)
    return fell

def _EB_to_QU(EB_maps: np.ndarray, lmax: int, **kwargs) -> np.ndarray:
    """
    Convert (T)EB maps to (T)QU maps.
    
    Parameters
    ----------
        EB_maps : np.ndarray
            Input (T)EB maps of shape (3, n_pixels) or (2, n_pixels).
        lmax : int
            Maximum multipole to consider for the conversion.
    
    Returns
    -------
        np.ndarray
            Converted (T)QU maps of shape (2, n_pixels) or (3, n_pixels).
    """
    alms_ = hp.map2alm(EB_maps, lmax=lmax, pol=False, **kwargs)
    if EB_maps.shape[0] == 3:
        return hp.alm2map(alms_,hp.get_nside(EB_maps[0]),lmax=lmax,pol=True)
    else:
        return hp.alm2map(np.array([0.*alms_[0],alms_[0],alms_[1]]),hp.get_nside(EB_maps[0]),lmax=lmax,pol=True)[1:]

def _E_to_QU(E_map: np.ndarray, lmax: int, **kwargs) -> np.ndarray:
    """
    Convert E-mode map to QU maps.
    
    Parameters
    ----------
        E_map : np.ndarray
            Input E-mode map of shape (n_pixels,).
        lmax : int
            Maximum multipole to consider for the conversion.

    Returns
    -------
        np.ndarray
            Converted QU maps of shape (2, n_pixels).
    """

    alms_ = hp.map2alm(E_map, lmax=lmax, pol=False, **kwargs)
    return hp.alm2map(np.array([0.*alms_,alms_,0.*alms_]),hp.get_nside(E_map),lmax=lmax,pol=True)[1:]

def _B_to_QU(B_map: np.ndarray, lmax: int, **kwargs) -> np.ndarray:
    """
    Convert B-mode map to QU maps.
    
    Parameters
    ----------
        B_map : np.ndarray
            Input B-mode map of shape (n_pixels,).
        lmax : int
            Maximum multipole to consider for the conversion.

    Returns
    -------
        np.ndarray
            Converted QU maps of shape (2, n_pixels).
    """
    alms_ = hp.map2alm(B_map, lmax=lmax, pol=False, **kwargs)
    return hp.alm2map(np.array([0.*alms_,0.*alms_,alms_]),hp.get_nside(B_map),lmax=lmax,pol=True)[1:]

def _QU_to_EB(QU_maps: np.ndarray, lmax: int, **kwargs) -> np.ndarray:
    """
    Convert (T)QU maps to (T)EB maps.
    
    Parameters
    ----------
        QU_maps : np.ndarray
            Input (T)QU maps of shape (3, n_pixels) or (2, n_pixels).
        lmax : int
            Maximum multipole to consider for the conversion.

    Returns
    -------
        np.ndarray
            Converted (T)EB maps of shape (3, n_pixels) or (2, n_pixels).
    """
    if QU_maps.shape[0] == 3:
        alms_ = hp.map2alm(QU_maps, lmax=lmax, pol=True, **kwargs)
        return hp.alm2map(alms_,hp.get_nside(QU_maps[0]),lmax=lmax,pol=False)
    elif QU_maps.shape[0] == 2:
        alms_ = hp.map2alm(np.array([0. * QU_maps[0], QU_maps[0], QU_maps[1]]), lmax=lmax, pol=True, **kwargs)
        return hp.alm2map(np.array([0.*alms_[0],alms_[1],alms_[2]]),hp.get_nside(QU_maps[0]),lmax=lmax,pol=True)[1:]

def _bl_from_fwhms(fwhm_out: float, fwhm_in: float, lmax: int) -> np.ndarray:
    """
    Compute beam transfer function to correct for input gaussian beam and convolve for output beam.
    
    Parameters
    ----------
        fwhm_out : float
            Full width at half maximum of the output beam in arcminutes.
        fwhm_in : float
            Full width at half maximum of the input beam in arcminutes.
        lmax : int
            Maximum multipole to consider for the beam transfer function.

    Returns
    -------
        np.ndarray
            Beam transfer function of shape (lmax+1, 3) for T, E, B modes.    
    """

    bl_in = hp.gauss_beam(np.radians(fwhm_in/60.), lmax=lmax,pol=True)
    bl_out = hp.gauss_beam(np.radians(fwhm_out/60.), lmax=lmax,pol=True)
    return bl_out / bl_in

def _bl_from_file(beam_path: str, channel: str, fwhm_out: float, input_beams: str, lmax: int) -> np.ndarray:
    """
    Load beam transfer function from file and compute output/input ratio.
    
    Parameters
    ----------
        beam_path : str
            Path to the beam transfer function file.
        channel : str
            Channel identifier for the beam file.
        fwhm_out : float
            Full width at half maximum of the output beam in arcminutes.
        input_beams : str
            Type of input beams, either "file_l" or "file_lm".
        lmax : int
            Maximum multipole to consider for the beam transfer function.

    Returns
    -------
        np.ndarray
            Beam transfer function of shape (lmax+1, 3) for T, E, B modes.
    """
    beam_file = f"{beam_path}_{channel}.fits"
    
    bl_in = _get_beam_from_file(beam_file,lmax,symmetric_beam=(input_beams != "file_lm"))
    bl_out_l = hp.gauss_beam(np.radians(fwhm_out/60.), lmax=lmax,pol=True)

    if input_beams == "file_l":
        return (bl_out_l[:,:3]) / (bl_in[:,:3])
    elif input_beams == "file_lm":
        bl_out = np.zeros((hp.Alm.getsize(lmax),3),dtype=complex)
        for ell in range(0, lmax + 1):
            idx_lmax = np.array([hp.Alm.getidx(lmax, ell, m) for m in range(ell + 1)])
            bl_out[idx_lmax,:] = np.tile(bl_out_l[ell,:3], (ell+1, 1))
        return bl_out / (bl_in[:,:3]) 

def  _get_beam_from_file(beam_file: str, lmax: int, symmetric_beam: bool = True) -> np.ndarray:
    """
    Read beam transfer function from FITS file.
    
    Parameters
    ----------
        beam_file : str
            Path to the beam transfer function file.
        lmax : int
            Maximum multipole to consider for the beam transfer function.
        symmetric_beam : bool, optional
            If True, assume the beam is symmetric. Default is True.

    Returns
    -------
        np.ndarray
            Beam transfer function of shape (lmax+1, 3) for T, E, B modes.
    
    """
    hdul = fits.open(beam_file)
    bl_file = np.column_stack([hdul[1].data[col].astype(str).astype(float) for col in hdul[1].data.names]).squeeze()

    if bl_file.ndim != 2 or bl_file.shape[1] not in [3, 4]:
        raise ValueError("Beam file must be 2-dimensional and have 3 or 4 columns (for T, E, B and EB).")

    if symmetric_beam:
        if lmax >= bl_file.shape[0]:
            raise ValueError("Insufficient multipoles in beam file.")
        return bl_file[:lmax+1]

    if lmax > hp.Alm.getlmax(bl_file.shape[0]):
        raise ValueError("The provided asymmetric beam file does not have enough values for the given alm.")

    bl = np.zeros((hp.Alm.getsize(lmax),bl_file.shape[1]),dtype=complex)
    lmax_file = hp.Alm.getlmax(bl_file.shape[0])
    for ell in range(0, lmax + 1):
        idx_lmax = np.array([hp.Alm.getidx(lmax, ell, m) for m in range(ell + 1)])
        idx_lmax_file = np.array([hp.Alm.getidx(lmax_file, ell, m) for m in range(ell + 1)])
        bl[idx_lmax,:] = bl_file[idx_lmax_file,:]
    return bl

def taper_beam_to_zero(l, ell0, ell1):
    """
    Taper function to zero between ell0 and ell1 for beams.
    - for ell <= ell0: unchanged (W=1)
    - for ell0 < ell <= ell1: smoothly tapered
    - for ell > ell1: zero

    Parameters
    ----------
    l : ndarray
        array of multipole moments
    ell0 : int
        multipole below which the beam is unchanged
    ell1 : int
        multipole above which the beam is set to zero
    
    Returns
    ----------
    taper_window : ndarray
        multiplicative window to be applied to bl (1 for ell<=ell0, 0 for ell>ell1)
    """
    if ell1 <= ell0:
        raise ValueError("ell1 must be > ell0")

    # window initialization
    W = np.zeros_like(l, dtype=float)
    mask_lo = l <= ell0
    mask_mid = (l > ell0) & (l <= ell1)
    mask_hi = l > ell1

    W[mask_lo] = 1.0
    W[mask_hi] = 0.0

    if np.any(mask_mid):
        phase = (l[mask_mid] - ell0) / (ell1 - ell0)   # in (0,1]
        W[mask_mid] = 0.5 * (1.0 + np.cos(np.pi * phase))

    return W

def _get_bandwidths(config: Configs, good_channels: np.ndarray) -> Optional[Union[np.ndarray, List[str]]]:
    """
    Retrieves the bandwidths for the specified channels based on the configuration.

    Parameters
    ----------
        config : Configs
            Configuration object containing instrument settings.
        good_channels : np.ndarray
            Indices of the channels to retrieve bandwidths for.
    
    Returns
    -------
        Optional[np.ndarray, List[str]]
            Bandwidths as a numpy array if `bandpass_integrate` is True, otherwise None.
            If `path_bandpasses` is set in the instrument configuration, returns a list of file paths.
    """

    # Bandwidth setup    
    if config.bandpass_integrate:
        if hasattr(config.instrument, "path_bandpasses"):
            bandwidths = [
                config.instrument.path_bandpasses + f"_{config.instrument.channels_tags[i]}.npy"
                for i in good_channels] 
        else: 
            bandwidths = np.array(config.instrument.bandwidth)[good_channels]
        return bandwidths
    else:
        return None

def _get_local_cov(
    input_maps: np.ndarray,
    lmax: int,
    ilc_bias: float,
    fwhm_out: float,
    b_ell: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    reduce_bias: bool = False,
    input_maps_2: Optional[np.ndarray] = None,
    variance: bool = False
) -> np.ndarray:
    """
    Compute the local covariance matrix between input maps.

    Parameters
    ----------
        input_maps: np.ndarray
            Input maps of shape (n_channels, n_pixels).
        lmax: int 
            Maximum multipole used in the harmonic space.
        ilc_bias: float
            Term used for smoothing scale estimation to optimally sample the covariance.
        fwhm_out: float
            Full width at half maximum of the input maps in arcminutes.
        b_ell: np.ndarray, optional
            Needlet transfer function. Defaults: unity.
        mask: np.ndarray, optional
            Optional sky mask. If provided, the covariance is computed only on the non-masked pixels.
        reduce_bias: bool
            If True, apply ILC bias-reduction technique.
        input_maps_2: np.ndarray, optional
            Second set of maps for computing cross-covariance.
        variance: bool
            If True, compute the variance instead of the power.
            Default is False.        

    Returns
    -------
        cov: np.ndarray
            Covariance matrices of shape (n_channels, n_channels, n_pixels).
    """
    if not isinstance(b_ell, np.ndarray):
        b_ell = np.ones(lmax+1)

    n_channels = input_maps.shape[0]
    n_pixels = input_maps.shape[1]

    fwhm_stat = get_fwhm_for_cov(lmax, n_pixels, n_channels, ilc_bias, fwhm_out, b_ell=b_ell)

#    nside_cov = hp.npix2nside(n_pixels) if mask is not None else int(np.min([hp.npix2nside(n_pixels), 128]))
    pixs_per_fwhm = 6. #if domain == "needlet" else 60.
    nside_cov = nside_from_fwhm(fwhm_stat, unit="rad", pixs_per_fwhm = pixs_per_fwhm)
    nside_cov = int(np.min([hp.npix2nside(n_pixels), nside_cov]))

    cov = np.zeros((n_channels, n_channels, 12 * nside_cov ** 2))

    for i in range(n_channels):
        for k in range(i,n_channels):
            map1 = input_maps[i] * mask if mask is not None else input_maps[i]
            map2 = (input_maps_2[k] if input_maps_2 is not None else input_maps[k])
            map2 = map2 * mask if mask is not None else map2

            cov[i,k] = cov_map = _get_local_cov_(
                map1, map2, fwhm_stat, nside_cov, reduce_bias=reduce_bias, variance=variance
            )

    for i in range(n_channels):
        for k in range(i):
            cov[i,k]=cov[k,i]
            
    return cov

def _get_local_cov_old(
    map1: np.ndarray,
    map2: np.ndarray,
    fwhm_stat: float,
    nside_covar: Optional[int] = None,
    reduce_bias: bool = False
) -> np.ndarray:
    """
    Compute local covariance between two maps.

    Parameters
    ----------
        map1: np.ndarray
            First input map.
        map2: np.ndarray
            Second input map.
        fwhm_stat: float
            Full width at half maximum for smoothing in radians.
        nside_covar: int, optional
            Nside resolution of output covariance. If None, it is set to Nside of map1 and map2
        reduce_bias: bool
            Apply smoothing bias reduction if True.

    Returns
    -------
        np.ndarray
            Covariance.
    """
    if hp.get_nside(map1) != hp.get_nside(map2):
        raise ValueError("Input maps must have the same HEALPix resolution.")

    product_map = map1 * map2
    nside = hp.get_nside(product_map)
    if nside_covar is None:
        nside_covar = nside

    # Degrade resolution for faster smoothing
    nside_out = max(1, nside // 4)
    stat_map = hp.ud_grade(product_map, nside_out = nside_out, order_in = 'RING', order_out = 'RING')
    
    # Compute alm
    lmax_stat = 2 * nside_out #3 * nside_out - 1 # 
    alm_s = hp.map2alm(stat_map, lmax=lmax_stat, iter=1, use_weights=True)# iter=0?

    # Get beam for smoothing covariance
    bl_stat = hp.gauss_beam(fwhm_stat, lmax_stat)

    # If reduce_bias is True, apply bias reduction technique
    if reduce_bias:
        thetas = np.arange(0,np.pi,0.002)
        beam_stat = hp.bl2beam(bl_stat, thetas)
        theta_ = 0.5 * fwhm_stat
        dist = 0.5 * (np.tanh(30 * (thetas - 0.3 * theta_)) - np.tanh(30 * (thetas - 3 * theta_)))
        dist /= np.max(dist)
        dist[np.argmax(dist):]=1.
        bl_stat = hp.beam2bl(dist * beam_stat,thetas, lmax = lmax_stat)

    alm_s = hp.almxfl(alm_s, bl_stat)
    
    return hp.alm2map(alm_s, nside_covar, lmax=lmax_stat)


def _get_local_cov_(
    map1: np.ndarray,
    map2: np.ndarray,
    fwhm_stat: float,
    nside_covar: Optional[int] = None,
    reduce_bias: bool = False,
    variance: bool = False
) -> np.ndarray:
    """
    Compute local covariance between two maps using full resolution smoothing
    without intermediate resolution degradation.

    Parameters
    ----------
        map1: np.ndarray
            First input map.
        map2: np.ndarray 
            Second input map.
        fwhm_stat: float
            Smoothing factor.
        nside_covar: int, optional
            Nside resolution of output covariance. If None, it is set based on fwhm_stat.
        reduce_bias: bool
            Apply smoothing bias reduction if True.
        variance: bool
            If True, compute variance instead of power of the product map.

    Returns
    -------
        np.ndarray
            Smoothed covariance map at specified resolution.
    """

    if hp.get_nside(map1) != hp.get_nside(map2):
        raise ValueError("Input maps must have the same HEALPix resolution.")

    npix = map1.size
    nside = hp.get_nside(map1)
    if nside_covar is None:
        nside_covar = nside_from_fwhm(fwhm_stat, unit="rad", pixs_per_fwhm = 3.)
        nside_covar = int(np.min([nside, nside_covar]))
    lmax_stat = 2 * nside_covar #3 * nside_out - 1 # 

    if variance:
        map1 -= hp.smoothing(map1, fwhm=fwhm_stat, lmax=lmax_stat, pol=False, verbose=False)
        map2 -= hp.smoothing(map2, fwhm=fwhm_stat, lmax=lmax_stat, pol=False, verbose=False)

    product_map = map1 * map2
    # Compute alm
    if 2 * nside_covar < nside:
        product_map = hp.ud_grade(product_map, nside_out = 2 * nside_covar, order_in = 'RING', order_out = 'RING')
    alm_s = hp.map2alm(product_map, lmax=lmax_stat, iter=1, use_weights=True)# iter=0?

    # Get beam for smoothing covariance
    bl_stat = hp.gauss_beam(fwhm_stat, lmax_stat)
    # If reduce_bias is True, apply bias reduction technique
    if reduce_bias:
        thetas = np.arange(0,np.pi,0.002)
        beam_stat = hp.bl2beam(bl_stat, thetas)
        theta_ = 0.5 * fwhm_stat
        dist = 0.5 * (np.tanh(30 * (thetas - 0.3 * theta_)) - np.tanh(30 * (thetas - 3 * theta_)))
        dist /= np.max(dist)
        dist[np.argmax(dist):]=1.
        bl_stat = hp.beam2bl(dist * beam_stat,thetas, lmax = lmax_stat)

    alm_s = hp.sphtfunc.almxfl(alm_s, bl_stat)

    return hp.alm2map(alm_s, nside_covar, lmax=lmax_stat) 

def get_fwhm_for_cov(lmax, n_pixels, n_channels, ilc_bias, fwhm_out, b_ell=None):
    """
    Compute the FWHM required for smoothing the covariance matrix.

    Parameters
    ----------
        lmax: int
            Maximum multipole.
        n_pixels: int
            Number of pixels in the map.
        n_channels: int
            Number of frequency channels.
        ilc_bias: float
            ILC bias parameter.
        b_ell: np.ndarray, optional
            Needlet transfer function. Defaults to unity.
    
    Returns
        float
            FWHM in radians.
    """

    if b_ell is None:
        b_ell = np.ones(lmax + 1)

    beam = hp.gauss_beam(np.deg2rad(fwhm_out/60.),lmax=lmax,pol=False)

    nmodes_band = np.sum((2. * np.arange(lmax + 1) + 1.) * (b_ell ** 2) * (beam ** 2))
    pps = np.sqrt(float(n_pixels) * float(n_channels-1) / (ilc_bias * nmodes_band) )
    pix_size = np.sqrt(4.0 * np.pi / n_pixels)

    return pps * pix_size

def merge_dicts(d: Union[List[Dict[Any, Any]], Dict[Any, Any]]) -> Dict[Any, Any]:
    """
    Merge a list of single-key dictionaries into one dictionary, or return the dictionary itself.

    Parameters:
    ----------
        d: dictionary or a list of dictionaries
            If a list, each dict is expected to have exactly one key.

    Returns
    -------
        Dict[Any, Any]
            A single merged dictionary.

    Raises
    -------
        ValueError: 
            If input is not a dict or list of single-key dicts.

    """

    if isinstance(d, list) and all(isinstance(item, dict) for item in d):
        if all(len(item) == 1 for item in d):
            merged_dict = {}
            for item in d:
                merged_dict.update(item)
            return merged_dict
        elif len(d) == 1 and isinstance(d[0], dict):
            return d[0]
    elif isinstance(d, dict):
        return d
    else:
        raise ValueError("Input must be a list of dictionaries or a single dictionary")
          
def obj_to_array(obj: SimpleNamespace, return_attributes: bool = False) -> Union[np.ndarray, List[str]]:
    """
    Convert a SimpleNamespace object with specified attributes into a numpy array.

    Attributes are stacked along the last axis.

    Parameters
    ----------
        obj
            SimpleNamespace with attributes like "total", "fgds", etc.
        return_attributes: bool
            If True, return the list of attributes of the object instead of the numpy array.
            Default is False.

    Returns
    --------
        np.ndarray
            A numpy array stacking the specified attributes if `return_attributes` is False.
        List[str]
            List of attribute names if `return_attributes` is True.

    Raises
    -------
        ValueError: If input is not a SimpleNamespace.
    """
    if not isinstance(obj, SimpleNamespace):
        raise ValueError("Input must be a SimpleNamespace object.")
    
    default_attributes = [
        "total", "total_split1", "total_split2", "fgds", "noise", "noise_split1", "noise_split2", "nuisance", "cmb", "dust", "synch", "ame",
        "co", "freefree", "cib", "tsz", "ksz", "radio_galaxies"
    ]

    obj_attributes = list(vars(obj).keys())

    used_attributes = [attr for attr in default_attributes if hasattr(obj, attr)]

    additional_attributes = [attr for attr in obj_attributes if attr not in default_attributes and hasattr(obj, attr)]

    all_used_attributes = used_attributes + additional_attributes

    array = [getattr(obj, attr) for attr in all_used_attributes if hasattr(obj, attr)]
    array = np.array(array)

    if return_attributes:
        return all_used_attributes
    else:
        if array.ndim == 3:
            return np.transpose(array, axes=(1,2,0))
        elif array.ndim == 4:
            return np.transpose(array, axes=(1,2,3,0))
    
def obj_out_to_array(obj: SimpleNamespace) -> np.ndarray:
    """
    Convert a SimpleNamespace object with output-related attributes to a numpy array.

    Parameters
    ----------
        obj
            SimpleNamespace with attributes like "output_total", "noise_residuals", etc.

    Returns
    --------
        np.ndarray
            A numpy array stacking the specified attributes.

    Raises
    -------
        ValueError: If input is not a SimpleNamespace.
    """
    if not isinstance(obj, SimpleNamespace):
        raise ValueError("Input must be a SimpleNamespace object.")

    default_attributes = [
        "output_total","output_total_split1","output_total_split2", "noise_residuals", "noise_split1_residuals", "noise_split2_residuals", "fgds_residuals",
        "output_cmb", "fgres_templates", "fgres_templates_noise",
        "fgres_templates_fgds"
    ]

    obj_attributes = list(vars(obj).keys())

    used_attributes = [attr for attr in default_attributes if hasattr(obj, attr)]

    additional_attributes = [attr for attr in obj_attributes if attr not in default_attributes and hasattr(obj, attr)]

    all_used_attributes = used_attributes + additional_attributes
    
    array = [getattr(obj, attr) for attr in all_used_attributes if hasattr(obj, attr)]
    
    return np.array(array)
    
def array_to_obj(array: np.ndarray, attr_list: List[str]) -> SimpleNamespace:
    """
    Convert a numpy array back to a SimpleNamespace object with specific attributes,
    based on attributes present in the reference object.

    Parameters
    ----------
        array: np.ndarray
            Numpy array where the last dimension corresponds to attributes.
        attr_list: List[str]
            List of attributes to assign from the array slices.

    Returns
    --------
        SimpleNamespace 
            SimpleNamespace with attributes set from array slices.
    """
    new_obj = SimpleNamespace()

    if len(attr_list) != array.shape[-1]:
        raise ValueError("Length of attr_list must match the last dimension of the array.")

    for idx, attr in enumerate(attr_list):
        setattr(new_obj, attr, array[..., idx])

    return new_obj

def _slice_data(data: SimpleNamespace, field_in: str, field_out: str) -> SimpleNamespace:
    """
    Slice data fields from a SimpleNamespace object according to field_in and field_out.

    Parameters
    ----------
        data: SimpleNamespace
            SimpleNamespace with input attributes.
        field_in: str
            String representing input field categories.
        field_out: str
            String representing fields to include.

    Returns
    --------
        SimpleNamespace
            SimpleNamespace containing sliced arrays.
    """

    data_out = SimpleNamespace()
    for attr_name in vars(data):
        setattr(data_out, attr_name, [])
    
    for idx, field in enumerate(field_in):
        if field in field_out:
            for attr_name in vars(data):
                getattr(data_out, attr_name).append(getattr(data, attr_name)[:,idx])

    for attr_name in vars(data_out):
        if np.array(getattr(data_out, attr_name)).shape[0]==1:
            setattr(data_out, attr_name, np.squeeze(np.transpose(np.array(getattr(data_out, attr_name)), axes=(1,0,2)), axis=1))        
        else:
            setattr(data_out, attr_name, np.transpose(np.array(getattr(data_out, attr_name)), axes=(1,0,2)))

    return data_out

def _slice_outputs(outputs: SimpleNamespace, field_in: str, field_out: str) -> SimpleNamespace:
    """
    Slice output fields from a SimpleNamespace object depending on requested fields.

    Parameters
    ----------
        outputs: SimpleNamespace
            SimpleNamespace with output attributes as arrays or lists.
        field_in: str
            Input fields.
        field_out: str
            Target fields to extract.

    Returns
    --------
        SimpleNamespace
            SimpleNamespace with sliced output arrays.
    """
    if field_out in ["QU_E", "QU_B"]:
        return outputs
    else:
        data_out = SimpleNamespace()
        for attr_name in vars(outputs):
            setattr(data_out, attr_name, [])

        for idx, field in enumerate(field_in):
            if field in field_out:
                for attr_name in vars(outputs):
                    getattr(data_out, attr_name).append(getattr(outputs, attr_name)[idx])

        for attr_name in vars(data_out):
            if np.array(getattr(data_out, attr_name)).shape[0]==1:
                setattr(data_out, attr_name, np.squeeze(np.array(getattr(data_out, attr_name)), axis=0))        
            else:
                setattr(data_out, attr_name, np.array(getattr(data_out, attr_name)))

        return data_out

def _slice_data_for_cls(data: SimpleNamespace, field_in: str, field_out: str) -> SimpleNamespace:
    """
    Perform specialized slicing of data for power spectrum calculations based on input and output field types.

    Parameters
    ----------
        data: SimpleNamespace
            SimpleNamespace with data arrays.
        field_in: str
            Input field type (e.g., "TQU", "TEB", "EB").
        field_out: str
            Desired output field type.

    Returns
    --------
        SimpleNamespace
            SimpleNamespace with appropriately sliced data.
    """

    if field_in == "TQU":
        if field_out in ["E", "B", "EB"]:
            data = _slice_data(data, field_in, "QU")
        elif field_out == "T":
            data = _slice_data(data, field_in, "T")
    elif field_in == "TEB":
        if field_out in ["T", "E", "B", "TE", "TB", "EB"]:
            data = _slice_data(data, field_in, field_out)
    elif field_in == "EB":
        if field_out in ["E", "B"]:
            data = _slice_data(data, field_in, field_out)
    return data

def _map2alm_kwargs(**kwargs) -> Dict[str, Any]:
    """
    Filter and create keyword arguments suitable for `hp.map2alm` from arbitrary kwargs.

    Allowed keys: iter, mmax, use_weights, datapath, use_pixel_weights.

    Parameters
    ----------
        kwargs: 
            Arbitrary keyword arguments.

    Returns
    -------
        map2alm_kwargs: Dict[str, Any]
            Dictionary containing only keys allowed by hp.map2alm.
    """
    map2alm_kwargs = {}
    # Define the allowed keywords for each function
    allowed_map2alm_kwargs = {"iter", "mmax", "use_weights", "datapath", "use_pixel_weights"}
    # Distribute kwargs
    for key, value in kwargs.items():
        if key in allowed_map2alm_kwargs:
            map2alm_kwargs[key] = value
    return map2alm_kwargs

def _format_nsim(nsim: Optional[Union[int, str]]) -> Optional[str]:
    """
    Format simulation number as zero-padded string or return None.

    Parameters
    ----------
        nsim: int, str, or None
            Simulation number as int, string, or None.

    Returns
    -------
        str or None
            Zero-padded 5-digit string if int, unchanged string if str, or None.

    Raises
    -------
        ValueError if nsim is not int, str, or None.
    """
    
    if nsim is None:
        return None
    if isinstance(nsim, int):
        return str(nsim).zfill(5)
    elif isinstance(nsim, str):
        return nsim
    else:
        raise ValueError("`nsim` must be an int, str, or None.")

def _log(message: str, verbose=False):
    """
    Utility for verbose logging.
    
    Parameters
    ----------
        message: str
            Message to log.
        verbose: bool, optional
            If True, print the message. Default is False.

    Returns
    -------
        None
    
    """
    if verbose:
        print(message)

def change_coord_mask(mask: np.ndarray, coord: List[str]) -> np.ndarray:
    """ Change coordinates of a HEALPIX mask

    Parameters
    ----------
        mask : np.ndarray
            mask(s) to be rotated. It can be a single mask or a stack of masks.
            Shape is (n_maps, n_pixels) or (n_pixels,).
        coord : list
            List of strings.
            First character is the coordinate system of mask, second character
            is the coordinate system of the output mask. As in HEALPIX, allowed
            coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

    Returns
    -------
        np.ndarray
            Mask(s) in the new coordinate system.
    """
    # Basic HEALPix parameters
    npix = mask.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))

    # Select the coordinate transformation
    rot = hp.Rotator(coord=reversed(coord))

    # Convert the coordinates
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)

    return mask[..., new_pix]

def nside_from_fwhm(fwhm, pixs_per_fwhm = 3., unit="arcmin"):
    """
    Compute the minimum HEALPix nside (power of 2) such that
    the pixel size is <= FWHM / pixs_per_fwhm.

    Parameters
    ----------
    fwhm : float
        Beam FWHM. 
    unit : {"arcmin", "deg", "rad"}, optional
        Unit of `fwhm`. Default is arcmin.
    pixs_per_fwhm : float, optional
        Number of pixels per FWHM. Default is 3.

    Returns
    -------
    nside : int
        Smallest power-of-two nside such that the typical
        HEALPix pixel size is <= FWHM/pixs_per_fwhm.
    """

    # --- Convert FWHM to radians ---
    if unit == "arcmin":
        fwhm_rad = np.deg2rad(fwhm / 60.0)
    elif unit == "deg":
        fwhm_rad = np.deg2rad(fwhm)
    elif unit == "rad":
        fwhm_rad = float(fwhm)
    else:
        raise ValueError("unit must be one of 'arcmin', 'deg', 'rad'")

    if fwhm_rad <= 0:
        raise ValueError("FWHM must be positive")

    nside_min_float = np.sqrt(np.pi / 3) * pixs_per_fwhm / fwhm_rad

    if nside_min_float < 1:
        return 1  # smallest valid nside

    k = int(np.ceil(np.log2(nside_min_float)))
    nside = 2**k

    return nside

def get_prefix(model, prefixes):
    """Get the prefix of a model from a list of prefixes.

    Parameters
    ----------
    model : str
        Model name.
    prefixes : List[str]
        List of possible prefixes.
    
    Returns
    -------
    str or None
        The matching prefix if found, otherwise None.
    """

    for p in prefixes:
        if model.startswith(p):
            return p
    return None

def get_fields_from_alms(input_alms: np.ndarray, field_out: str) -> List[str]:
    """
    Determine the fields present in the input alms based on their shape and the configuration parameters.

    Parameters
    ----------
        input_alms: np.ndarray
            Input alms array, which can have different shapes depending on the fields included.
        field_out: str
            Desired output field type (e.g., "T", "E", "B", "EB", "QU_E", "QU_B").
    
    Returns
        List[str]
            List of field identifiers (e.g., "T", "E", "B") corresponding to the input alms.
    """

    if input_alms.ndim == 4:
        if input_alms.shape[1] == 3:
            return ["T", "E", "B"]
        elif input_alms.shape[1] == 2:
            return ["E", "B"]
    elif input_alms.ndim == 3:
        if field_out in ["T", "E", "B"]:
            return [field_out]
        elif field_out in ["QU_E", "QU_B"]:
            return [field_out[-1]]

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]

