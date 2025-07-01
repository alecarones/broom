from scipy import constants
from astropy.cosmology import Planck18
from pysm3 import units as u
import numpy as np
from pysm3.models.dust import blackbody_ratio
from pysm3.utils import bandpass_unit_conversion, normalize_weights, trapz_step_inplace
from .needlets import _get_needlet_windows_
from typing import List, Union, Optional, Tuple, Dict
import sys

T_cmb = Planck18.Tcmb(0).value
h = constants.h
k = constants.k

def _get_CMB_SED(frequencies: List[float], units: str = "uK_CMB",
    bandwidths: Optional[Union[List[str], List[float]]] = None
) -> np.ndarray:
    """
    Function to compute the SED of the CMB in different units.

    Parameters
    ----------
        frequencies : list
            Frequencies in GHz.
        units : str, optional
            Units of the output SED. Options are "uK_CMB", "uK_RJ" and all their multiples (e.g., "mK_CMB", "mK_RJ") or "Jy_sr".
            Default is "uK_CMB".
        bandwidths : list or None, optional
            If list of str, it should contain the paths to npy files with frequency bands and response.
            If list of floats, it should contain the relative bandwidths in GHz for each frequency channel.
            If None, the SED is computed at the frequencies without assuming any bandwidth.

    Returns
    -------
        np.ndarray
            SED of the CMB in the specified units.
    """

    if units[-3:] == "CMB":
        return (u.Unit(units)).to(u.uK_CMB) * np.ones(len(frequencies))

    if bandwidths is not None:
        sed = []
        for idx_freq, frequency in enumerate(frequencies):
            if isinstance(bandwidths[0], str):
                freqs_band, weights = np.load(bandwidths[idx_freq])
            else:
                freq_min = frequency * (1 - ( bandwidths[idx_freq] / 2 ))
                freq_max = frequency * (1 + ( bandwidths[idx_freq] / 2 ))
                steps = int(freq_max - freq_min + 1)
                freqs_band = np.linspace(freq_min, freq_max, steps)
                weights = np.ones_like(freqs_band) # The tophat is defined in intensity units (Jy/sr)
            weights_rj = normalize_weights(freqs_band, weights)

            sed_band = np.zeros(1)
            for i, (freq, _weight) in enumerate(zip(freqs_band, weights_rj)):
                sed_ = u.uK_CMB.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freq * u.GHz))
                trapz_step_inplace(freqs_band, weights_rj, i, sed_, sed_band)

            if units != "Jy_sr":
                sed.append((sed_band * bandpass_unit_conversion(freqs_band * u.GHz, weights, getattr(u, units))).value[0])
            else:
                sed.append((sed_band * bandpass_unit_conversion(freqs_band * u.GHz, weights, u.Jy / u.sr)).value[0])

        return np.array(sed)

    else:
        if units[-2:] == "RJ":
            return u.uK_CMB.to(u.Unit(units), equivalencies=u.cmb_equivalencies(frequencies * u.GHz))
        elif units == "Jy_sr":
            return u.uK_CMB.to((u.Jy / u.sr), equivalencies=u.cmb_equivalencies(frequencies * u.GHz))

def _get_moments_SED(
    frequencies: List[float],
    moms_list: List[str],
    beta_d: float = 1.54,
    T_d: float = 20.0,
    beta_s: float = -3.0,
    nu_ref_d: float = 353.0,
    nu_ref_s: float = 30.0,
    units: str = "uK_CMB",
    bandwidths: Optional[Union[List[str], List[float]]] = None
) -> np.ndarray:
    """
    Function to compute the SED of the moments in different units.

    Parameters
    ----------
        frequencies : list
            Frequencies observed in GHz.
        moms_list : list
            List of moments to compute. It can include
            - '0d': zeroth moment of modified black body (thermal dust)
            - '0s': zeroth moment of power law (synchrotron)
            - '1bd': first derivative of modified black body wrt beta
            - '1bs': first derivative of power law wrt beta
            - '1Td': first derivative of modified black body wrt temperature
            - '2bd': second derivative of modified black body wrt beta
            - '2bs': second derivative of power law wrt beta
            - '2Td': second derivative of modified black body wrt temperature
            - '2bdTd': second derivative of modified black body wrt beta and temperature
            - '2Tdbd': second derivative of modified black body wrt temperature and beta
        beta_d : float, optional
            Spectral index of the modified black body (thermal dust). It refers to muK_RJ units SED.
            Default is 1.54.
        T_d : float, optional
            Temperature of the modified black body (thermal dust) in Kelvin. It refers to muK_RJ units SED. 
            Default is 20.
        beta_s : float, optional
            Spectral index of the power law (synchrotron). It refers to muK_RJ units SED. 
            Default is -3.
        nu_ref_d : float, optional
            Reference frequency for the modified black body (thermal dust) in GHz. 
            Default is 353.
        nu_ref_s : float, optional
            Reference frequency for the power law (synchrotron) in GHz.
            Default is 30.
        units : str, optional
            Units of the output SED. Options are "uK_CMB", "uK_RJ", or "Jy_sr" and all their multiples (e.g., "mK_CMB", "mK_RJ", "mJy_sr").
        bandwidths : list or None, optional
            If list of str, it should contain the paths to npy files with frequency bands and response.
            If list of floats, it should contain the relative bandwidths in GHz for each frequency channel.
            If None, the SED is computed at the frequencies without assuming any bandwidth.
    
    Returns
    -------
        np.ndarray
            SED of the moments in the specified units. The shape is (len(moms_list), len(frequencies)).
    """
    
    moments_funcs = {
    '0d': _mbb_0,
    '0s': _pl_0,
    '1bd': _mbb_1b,
    '1bs': _pl_1b,
    '1Td': _mbb_1T,
    '2bd': _mbb_2b,
    '2bs': _pl_2b,
    '2Td': _mbb_2T,
    '2bdTd': _mbb_2bT,
    '2Tdbd': _mbb_2bT,
    }

    all_seds = []
    for mom in moms_list:
        sed = []

        for idx_freq, frequency in enumerate(frequencies):
            if bandwidths is not None:
                if isinstance(bandwidths[0], str):
                    freqs_band, weights = np.load(bandwidths[idx_freq])
                else:
                    freq_min = frequency * (1 - ( bandwidths[idx_freq] / 2 ))
                    freq_max = frequency * (1 + ( bandwidths[idx_freq] / 2 ))
                    steps = int(freq_max - freq_min + 1)
                    freqs_band = np.linspace(freq_min, freq_max, steps)
                    weights = np.ones_like(freqs_band) # The tophat is defined in intensity units (Jy/sr)
                weights_rj = normalize_weights(freqs_band, weights)

                sed_band = np.zeros(1)
                for i, (freq, _weight) in enumerate(zip(freqs_band, weights_rj)):
                    if 'd' in mom:
                        sed_ = moments_funcs[mom](freq, beta_d, T_d, nu_ref = nu_ref_d)
                    elif 's' in mom:
                        sed_ = moments_funcs[mom](freq, beta_s, nu_ref = nu_ref_s)
                    trapz_step_inplace(freqs_band, weights_rj, i, sed_, sed_band)          
            else:
                if 'd' in mom:
                    sed_band = moments_funcs[mom](frequency, beta_d, T_d, nu_ref = nu_ref_d)
                elif 's' in mom:
                    sed_band = moments_funcs[mom](frequency, beta_s, nu_ref = nu_ref_s)
                freqs_band = frequency
                weights = None

            if units != "Jy_sr":
                if units[-3:] == "CMB":
                    sed.append((sed_band * bandpass_unit_conversion(freqs_band * u.GHz, weights, getattr(u, units))).value[0])
                elif units[-2:] == "RJ":
                    sed.append((sed_band * bandpass_unit_conversion(freqs_band * u.GHz, weights, getattr(u, units))).value)
            else:
                sed.append((sed_band * bandpass_unit_conversion(freqs_band * u.GHz, weights, u.Jy / u.sr)).value[0])

        all_seds.append(sed)
    
    return np.array(all_seds)
   

def _mbb_0(nu, beta, T, nu_ref = 335.):
    """
    Function to compute the SED of a modified black body.

    Parameters
    ----------
        nu : float or np.ndarray
            Frequency in GHz.
        beta : float
            Spectral index of the modified black body.
        T : float
            Temperature in Kelvin.
        nu_ref : float, optional
            Reference frequency in GHz. Default is 335 GHz.
    
    Returns
    -------
        np.ndarray
            SED of the modified black body in CMB brightness units.
    """
    x = (h * nu * (1e+9)) / (k * T)
    return ((nu/nu_ref)**(beta+1)) / (np.exp(x)-1)  

def _mbb_1b(nu, beta, T, nu_ref = 335.):
    """
    Function to compute the SED of first derivative of modified black body wrt beta.

    Parameters
    ----------
        nu : float or np.ndarray
            Frequency in GHz.
        beta : float
            Spectral index of the modified black body.
        T : float
            Temperature in Kelvin.
        nu_ref : float, optional
            Reference frequency in GHz. Default is 335 GHz.
        
    Returns
    -------
        np.ndarray
            SED of the first derivative of modified black body wrt beta in CMB brightness units.
    """
    return np.log(nu/nu_ref) * _mbb_0(nu, beta, T, nu_ref = nu_ref)

def _mbb_1T(nu, beta, T, nu_ref = 335.):
    """
    Function to compute the SED of first derivative of modified black body wrt temperature.

    Parameters
    ----------
        nu : float or np.ndarray
            Frequency in GHz.
        beta : float
            Spectral index of the modified black body.
        T : float
            Temperature in Kelvin.
        nu_ref : float, optional
            Reference frequency in GHz. Default is 335 GHz.
    
    Returns
    -------
        np.ndarray
            SED of the first derivative of modified black body wrt temperature in CMB brightness units.
    """
    x = (h * nu * (1e+9)) / (k * T)
    return x * np.exp(x) * _mbb_0(nu, beta, T, nu_ref = nu_ref) / (T * (np.exp(x)-1))

def _mbb_2b(nu, beta, T, nu_ref = 335.):
    """
    Function to compute the SED of second derivative of modified black body wrt beta.

    Parameters
    ----------
        nu : float or np.ndarray
            Frequency in GHz.
        beta : float
            Spectral index of the modified black body.
        T : float
            Temperature in Kelvin.
        nu_ref : float, optional
            Reference frequency in GHz. Default is 335 GHz.

    Returns
    -------
        np.ndarray
            SED of the second derivative of modified black body wrt beta in CMB brightness units.
    """
    return ((np.log(nu/nu_ref))**2) * _mbb_0(nu, beta, T, nu_ref = nu_ref)

def _mbb_2T(nu, beta, T, nu_ref = 335.):
    """
    Function to compute the SED of second derivative of modified black body wrt temperature.

    Parameters
    ----------
        nu : float or np.ndarray
            Frequency in GHz.
        beta : float
            Spectral index of the modified black body.
        T : float
            Temperature in Kelvin.
        nu_ref : float, optional
            Reference frequency in GHz. Default is 335 GHz.

    Returns
    -------
        np.ndarray
            SED of the second derivative of modified black body wrt temperature in CMB brightness units.
    """
    x = (h * nu * (1e+9)) / (k * T)
    return ((x/np.tanh(x/2)) - 2.) * _mbb_1T(nu, beta, T, nu_ref = nu_ref) / T

def _mbb_2bT(nu, beta, T, nu_ref = 335.):
    """
    Function to compute the SED of second derivative of modified black body wrt beta and temperature.

    Parameters
    ----------
        nu : float or np.ndarray
            Frequency in GHz.
        beta : float
            Spectral index of the modified black body.
        T : float
            Temperature in Kelvin.
        nu_ref : float, optional
            Reference frequency in GHz. Default is 335 GHz.

    Returns
    -------
        np.ndarray
            SED of the second derivative of modified black body wrt beta and temperature in CMB brightness units.
    """
    return np.log(nu/nu_ref) * _mbb_1T(nu, beta, T, nu_ref = nu_ref)

def _pl_0(nu, beta, nu_ref = 30.):
    """
    Function to compute the SED of a power law.

    Parameters
    ----------
        nu : float or np.ndarray
            Frequency in GHz.
        beta : float
            Spectral index of the power law.
        nu_ref : float, optional
            Reference frequency in GHz. Default is 30 GHz.

    Returns
    -------
        np.ndarray
            SED of the power law in CMB brightness units.
    """
    return (nu/nu_ref)**(beta)

def _pl_1b(nu, beta, nu_ref = 30.):
    """
    Function to compute the SED of first derivative of power law wrt beta.

    Parameters
    ----------
        nu : float or np.ndarray
            Frequency in GHz.
        beta : float
            Spectral index of the power law.
        nu_ref : float, optional
            Reference frequency in GHz. Default is 30 GHz.

    Returns
    -------
        np.ndarray
            SED of the first derivative of power law wrt beta in CMB brightness units.
    """
    return np.log(nu/nu_ref) * _pl_0(nu, beta, nu_ref = nu_ref)

def _pl_2b(nu, beta, nu_ref = 30.):
    """
    Function to compute the SED of second derivative of power law wrt beta.

    Parameters
    ----------
        nu : float or np.ndarray
            Frequency in GHz.
        beta : float
            Spectral index of the power law.
        nu_ref : float, optional
            Reference frequency in GHz. Default is 30 GHz.

    Returns
    -------
        np.ndarray
            SED of the second derivative of power law wrt beta in CMB brightness units.
    """
    return ((np.log(nu/nu_ref))**2) * _pl_0(nu, beta, nu_ref = nu_ref)

def _standardize_cilc(compsep_run: dict, lmax: int) -> dict:
    """
    Function to standardize and validate the configuration dictionary for cILC component separation.
    It checks and sets the necessary parameters for the cILC method.

    Parameters
    ----------
        compsep_run : dict
            Configuration dictionary for the cILC component separation run.
        lmax : int
            Maximum multipole for the component separation.
    
    Returns
    -------
        compsep_run : dict
            Updated configuration dictionary with standardized parameters.
    """
    if compsep_run["domain"] == "needlet":
        if compsep_run["method"] in ["cilc","cpilc"]:
            nls_number = _get_needlet_windows_(compsep_run["needlet_config"], lmax).shape[0]
        elif compsep_run["method"] in ["c_ilc", "c_pilc"]:
            nls_number = len(compsep_run["special_nls"])
        elif compsep_run["method"] in ["mc_cilc"]:
            nls_number = _get_needlet_windows_(compsep_run["needlet_config"], lmax).shape[0] - len(compsep_run["special_nls"])

    if not "moments" in compsep_run["constraints"]:
        raise ValueError("A list of moments must be provided in the constraints dictionary.")
    else:
        if not isinstance(compsep_run["constraints"]["moments"], list):
            raise ValueError("The moments must be provided as a list.")

    dim_moms = len(compsep_run["constraints"]["moments"]) if isinstance(compsep_run["constraints"]["moments"][0], list) else 1
    if dim_moms == 1 and isinstance(compsep_run["constraints"]["moments"][0], list):
        compsep_run["constraints"]["moments"] = compsep_run["constraints"]["moments"][0]

    if (dim_moms > 1) and (compsep_run["domain"]=="pixel"):
        raise ValueError("If the domain is pixel, moments must be a 1d list.")

    if "deprojection" in compsep_run["constraints"]:
        if not isinstance(compsep_run["constraints"]["deprojection"], (int, float, list)):
            raise ValueError("The deprojection parameter must be a scalar or a list.")
        if isinstance(compsep_run["constraints"]["deprojection"], list):
            dim_depro = len(compsep_run["constraints"]["deprojection"]) if isinstance(compsep_run["constraints"]["deprojection"][0], list) else 1
            if dim_depro != dim_moms:
                raise ValueError("The list of deprojection coefficients must have the same dimension as the moments list.")

            if dim_depro == 1 and isinstance(compsep_run["constraints"]["deprojection"][0], list):
                compsep_run["constraints"]["deprojection"] = compsep_run["constraints"]["deprojection"][0]
            
            if dim_depro == 1:
                if len(compsep_run["constraints"]["deprojection"]) != len(compsep_run["constraints"]["moments"]):
                    raise ValueError("The list of deprojection coefficients must have the same dimension as the moments list.")
            else:
                for n in range(len(compsep_run["constraints"]["deprojection"])):
                    if len(compsep_run["constraints"]["deprojection"][n]) != len(compsep_run["constraints"]["moments"][n]):
                        raise ValueError("The list of deprojection coefficients must have the same dimension as the moments list.")
            
    if compsep_run["domain"] == "needlet":
        if dim_moms > 1:
            if dim_moms > nls_number:
                compsep_run["constraints"]["moments"] = compsep_run["constraints"]["moments"][:nls_number]
            elif dim_moms < nls_number:
                while len(compsep_run["constraints"]["moments"]) < nls_number:
                    (compsep_run["constraints"]["moments"]).append(compsep_run["constraints"]["moments"][-1])
        else:
            compsep_run["constraints"]["moments"] = [compsep_run["constraints"]["moments"] for _ in range(nls_number)]

    if not "deprojection" in compsep_run["constraints"]:
        if compsep_run["domain"] == "pixel":
            compsep_run["constraints"]["deprojection"] = [0. for _ in compsep_run["constraints"]["moments"]]
        elif compsep_run["domain"] == "needlet":
            compsep_run["constraints"]["deprojection"] = [[0. for _ in row] for row in compsep_run["constraints"]["moments"]]
    elif isinstance(compsep_run["constraints"]["deprojection"], (int, float)):
        if compsep_run["domain"] == "pixel":
            compsep_run["constraints"]["deprojection"] = [compsep_run["constraints"]["deprojection"] for _ in compsep_run["constraints"]["moments"]]
        elif compsep_run["domain"] == "needlet":
            compsep_run["constraints"]["deprojection"] = [[compsep_run["constraints"]["deprojection"] for _ in row] for row in compsep_run["constraints"]["moments"]]
    elif isinstance(compsep_run["constraints"]["deprojection"], list):
        if compsep_run["domain"] == "needlet":
            if dim_depro > 1:
                if dim_depro > nls_number:
                    compsep_run["constraints"]["deprojection"] = compsep_run["constraints"]["deprojection"][:nls_number]
                elif dim_depro < nls_number:
                    while len(compsep_run["constraints"]["deprojection"]) < nls_number:
                        (compsep_run["constraints"]["deprojection"]).append(compsep_run["constraints"]["deprojection"][-1])
            else:
                compsep_run["constraints"]["deprojection"] = [compsep_run["constraints"]["deprojection"] for _ in range(nls_number)]

    if not "beta_d" in compsep_run["constraints"]:
        compsep_run["constraints"]["beta_d"] = 1.54
    
    if not isinstance(compsep_run["constraints"]["beta_d"], (int, float, list)):
        raise ValueError("The beta_d parameter must be a scalar or a list.")
    else:
        if isinstance(compsep_run["constraints"]["beta_d"], list):
            dim_beta = len(compsep_run["constraints"]["beta_d"]) if isinstance(compsep_run["constraints"]["beta_d"][0], list) else 1
            if dim_beta == 1 and isinstance(compsep_run["constraints"]["beta_d"][0], list):
                compsep_run["constraints"]["beta_d"] = compsep_run["constraints"]["beta_d"][0]
            elif dim_beta > 1:
                raise ValueError("The beta_d parameter must be a scalar or a list of scalars.")

        if compsep_run["domain"] == "needlet":
            if isinstance(compsep_run["constraints"]["beta_d"], (int, float)):
                compsep_run["constraints"]["beta_d"] = [compsep_run["constraints"]["beta_d"] for _ in range(nls_number)]
            elif len(compsep_run["constraints"]["beta_d"]) < nls_number:
                while len(compsep_run["constraints"]["beta_d"]) < nls_number:
                    (compsep_run["constraints"]["beta_d"]).append(compsep_run["constraints"]["beta_d"][-1])
            elif len(compsep_run["constraints"]["beta_d"]) > nls_number:
                compsep_run["constraints"]["beta_d"] = compsep_run["constraints"]["beta_d"][:nls_number]
        elif compsep_run["domain"] == "pixel":
            if isinstance(compsep_run["constraints"]["beta_d"], list):
                raise ValueError("If the domain is pixel, beta_d must be a scalar.")

    if not "T_d" in compsep_run["constraints"]:
        compsep_run["constraints"]["T_d"] = 20.
    
    if not isinstance(compsep_run["constraints"]["T_d"], (int, float, list)):
        raise ValueError("The T_d parameter must be a scalar or a list.")
    else:
        if isinstance(compsep_run["constraints"]["T_d"], list):
            dim_T = len(compsep_run["constraints"]["T_d"]) if isinstance(compsep_run["constraints"]["T_d"][0], list) else 1
            if dim_T == 1 and isinstance(compsep_run["constraints"]["T_d"][0], list):
                compsep_run["constraints"]["T_d"] = compsep_run["constraints"]["T_d"][0]
            elif dim_T > 1:
                raise ValueError("The T_d parameter must be a scalar or a list of scalars.")

        if compsep_run["domain"] == "needlet":
            if isinstance(compsep_run["constraints"]["T_d"], (int, float)):
                compsep_run["constraints"]["T_d"] = [compsep_run["constraints"]["T_d"] for _ in range(nls_number)]
            elif len(compsep_run["constraints"]["T_d"]) < nls_number:
                while len(compsep_run["constraints"]["T_d"]) < nls_number:
                    (compsep_run["constraints"]["T_d"]).append(compsep_run["constraints"]["T_d"][-1])
            elif len(compsep_run["constraints"]["T_d"]) > nls_number:
                compsep_run["constraints"]["T_d"] = compsep_run["constraints"]["T_d"][:nls_number]
        elif compsep_run["domain"] == "pixel":
            if isinstance(compsep_run["constraints"]["T_d"], list):
                raise ValueError("If the domain is pixel, T_d must be a scalar.")
                
    if not "beta_s" in compsep_run["constraints"]:
        compsep_run["constraints"]["beta_s"] = -3.
    
    if not isinstance(compsep_run["constraints"]["beta_s"], (int, float, list)):
        raise ValueError("The beta_s parameter must be a scalar or a list.")
    else:
        if isinstance(compsep_run["constraints"]["beta_s"], list):
            dim_beta = len(compsep_run["constraints"]["beta_s"]) if isinstance(compsep_run["constraints"]["beta_s"][0], list) else 1
            if dim_beta == 1 and isinstance(compsep_run["constraints"]["beta_s"][0], list):
                compsep_run["constraints"]["beta_s"] = compsep_run["constraints"]["beta_s"][0]
            elif dim_beta > 1:
                raise ValueError("The beta_s parameter must be a scalar or a list of scalars.")

        if compsep_run["domain"] == "needlet":
            if isinstance(compsep_run["constraints"]["beta_s"], (int, float)):
                compsep_run["constraints"]["beta_s"] = [compsep_run["constraints"]["beta_s"] for _ in range(nls_number)]
            elif len(compsep_run["constraints"]["beta_s"]) < nls_number:
                while len(compsep_run["constraints"]["beta_s"]) < nls_number:
                    (compsep_run["constraints"]["beta_s"]).append(compsep_run["constraints"]["beta_s"][-1])
            elif len(compsep_run["constraints"]["beta_s"]) > nls_number:
                compsep_run["constraints"]["beta_s"] = compsep_run["constraints"]["beta_s"][:nls_number]
        elif compsep_run["domain"] == "pixel":
            if isinstance(compsep_run["constraints"]["beta_s"], list):
                raise ValueError("If the domain is pixel, beta_s must be a scalar.")

    return compsep_run

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]
