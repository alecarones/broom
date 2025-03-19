from scipy import constants
from astropy.cosmology import Planck18
from pysm3 import units as u
import numpy as np

T_cmb = Planck18.Tcmb(0).value
h = constants.h
k = constants.k

def _get_CMB_SED(frequencies, units="K_CMB"):
    if units[-3:] == "CMB":
        return np.ones(len(frequencies))
    elif units[-2:] == "RJ":
        return u.uK_CMB.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(frequencies * u.GHz))
    elif units == "Jy_sr":
        return u.uK_CMB.to((u.Jy / u.sr), equivalencies=u.cmb_equivalencies(frequencies * u.GHz))

#def _mbb_(frequencies, beta, T, units="K_CMB", bandpass_weights=None):
