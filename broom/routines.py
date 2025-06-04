import numpy as np
import healpy as hp
import mtneedlet as nl
from astropy.io import fits
import re
import os
from .configurations import Configs
from .leakage import purify_master, purify_recycling
from types import SimpleNamespace


def _alms_from_data(config: Configs, data, field_in, mask_in=None, data_type="maps", bring_to_common_resolution=True, pixel_window_in=False, **kwargs):
    fell = _get_ell_filter(2,config.lmax)

    if data_type == "maps":
        alms = _maps_to_alms(config, data, field_in, mask_in=mask_in, **kwargs)
    elif data_type == "alms":
        alms = _alms_to_alms(config, data)

    alms = _processing_alms(config, alms, bring_to_common_resolution=bring_to_common_resolution, pixel_window_in=pixel_window_in)    

    return alms

def _maps_to_alms(config: Configs, data, field_in, mask_in=None, **kwargs):
    fell = _get_ell_filter(2,config.lmax)

    if mask_in is None:
        mask_in = np.ones(data.total.shape[-1])
    elif not isinstance(mask_in, np.ndarray): 
        raise ValueError("Invalid mask. It must be a numpy array.")
    elif hp.get_nside(mask_in) != hp.npix2nside(data.total.shape[-1]):
            raise ValueError("Mask HEALPix resolution does not match data HEALPix resolution.")
    mask_in /= np.max(mask_in)
    mask_bin_in = np.ceil(mask_in)

    if config.leakage_correction is not None:
        if "_recycling" in config.leakage_correction:
            if "_iterations" in config.leakage_correction:
                iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
            else:
                iterations = 0
    
    alms = SimpleNamespace()

    if data.total.ndim == 2:
        for attr_name in vars(data):
            setattr(alms, attr_name, [])
            for i in range(getattr(data, attr_name).shape[0]):
                getattr(alms, attr_name).append(hp.almxfl(hp.map2alm((getattr(data, attr_name)[i] * mask_bin_in), lmax=config.lmax, pol=False, **kwargs),fell))
            setattr(alms, attr_name, np.array(getattr(alms, attr_name)))

    else:
        if data.total.shape[1]==3:
            for attr_name in vars(data):
                setattr(alms, attr_name, np.zeros((getattr(data, attr_name).shape[0], getattr(data, attr_name).shape[1], hp.Alm.getsize(config.lmax)), dtype=complex))
                for i in range(getattr(data, attr_name).shape[0]):
                    if (config.leakage_correction is None) or (field_in=="TEB") or (np.mean(mask_in**2)==1.):
                        getattr(alms, attr_name)[i] = hp.map2alm((getattr(data, attr_name)[i] * mask_bin_in), lmax=config.lmax, pol=field_in=="TQU", **kwargs)
                    else:
                        getattr(alms, attr_name)[i, 0] = hp.map2alm((getattr(data, attr_name)[i, 0] * mask_bin_in), lmax=config.lmax, **kwargs)
                        if config.leakage_correction == "mask_only":
                            getattr(alms, attr_name)[i, 1:] = hp.map2alm((getattr(data, attr_name)[i] * mask_in), lmax=config.lmax, pol=True, **kwargs)[1:]
                        elif "_purify" in config.leakage_correction:
                            getattr(alms, attr_name)[i, 1:] = purify_master(getattr(data, attr_name)[i, 1:], mask_in, config.lmax,purify_E=("E" in config.leakage_correction))
                        elif "_recycling" in config.leakage_correction:
                            getattr(alms, attr_name)[i, 1:] = purify_recycling(getattr(data, attr_name)[i, 1:], data.total[i,1:], mask_bin_in, config.lmax,purify_E=("E" in config.leakage_correction), iterations=iterations, **kwargs)
                    for j in range(getattr(alms, attr_name).shape[1]):
                        getattr(alms, attr_name)[i, j] = hp.almxfl(getattr(alms, attr_name)[i, j], fell)

        elif data.total.shape[1]==2:
            for attr_name in vars(data):
                if config.field_out in ["E", "B", "QU_E", "QU_B"]:
                    setattr(alms, attr_name, np.zeros((getattr(data, attr_name).shape[0], hp.Alm.getsize(config.lmax)), dtype=complex))
                else:
                    setattr(alms, attr_name, np.zeros((getattr(data, attr_name).shape[0], getattr(data, attr_name).shape[1], hp.Alm.getsize(config.lmax)), dtype=complex))
                for i in range(getattr(data, attr_name).shape[0]):
                    if (config.leakage_correction is None) or (np.mean(mask_in**2)==1.) or (field_in=="EB") or (config.field_out in ["E", "QU_E"] and ("E" not in config.leakage_correction)) or (config.field_out in ["B", "QU_B"] and ("B" not in config.leakage_correction)):
                        getattr(alms, attr_name)[i] = hp.map2alm(np.vstack((0. * getattr(data, attr_name)[i, 0], getattr(data, attr_name)[i])) * mask_bin_in, lmax=config.lmax, pol=field_in=="QU", **kwargs)[[1,2] if config.field_out in ["QU", "EB"] else 1 if config.field_out in ["E", "QU_E"] else 2]
                    else:
                        if config.leakage_correction == "mask_only":
                            getattr(alms, attr_name)[i] = hp.map2alm(np.vstack((0. * getattr(data, attr_name)[i, 0], getattr(data, attr_name)[i])) * mask_in, lmax=config.lmax, pol=True, **kwargs)[[1,2] if config.field_out in ["QU", "EB"] else 1 if config.field_out in ["E", "QU_E"] else 2]
                        elif "_purify" in config.leakage_correction:
                            alms_pure = purify_master(getattr(data, attr_name)[i], mask_in, config.lmax,purify_E=("E" in config.leakage_correction))
                            getattr(alms, attr_name)[i] = np.copy(alms_pure) if config.field_out in ["QU", "EB"] else np.copy(alms_pure[0]) if config.field_out in ["E", "QU_E"] else np.copy(alms_pure[1])
                        elif "_recycling" in config.leakage_correction:
                            alms_pure = purify_recycling(getattr(data, attr_name)[i], data.total[i], mask_bin_in, config.lmax,purify_E=("E" in config.leakage_correction), iterations=iterations, **kwargs)
                            getattr(alms, attr_name)[i] = np.copy(alms_pure) if config.field_out in ["QU", "EB"] else np.copy(alms_pure[0]) if config.field_out in ["E", "QU_E"] else np.copy(alms_pure[1])
                    if config.field_out in ["E", "B", "QU_E", "QU_B"]:
                        getattr(alms, attr_name)[i] = hp.almxfl(getattr(alms, attr_name)[i], fell)
                    else:
                        for j in range(getattr(alms, attr_name).shape[1]):
                            getattr(alms, attr_name)[i, j] = hp.almxfl(getattr(alms, attr_name)[i, j], fell)
                            
    return alms

def _alms_to_alms(config: Configs, data):
    fell = _get_ell_filter(2,config.lmax)
    alms = SimpleNamespace()

    if hp.Alm.getlmax(data.total.shape[-1]) == config.lmax:
        for attr_name in vars(data):
            if getattr(data, attr_name).ndim == 2 or (getattr(data, attr_name).ndim == 3 and getattr(data, attr_name).shape[1]==3) or (getattr(data, attr_name).ndim == 3 and getattr(data, attr_name).shape[1]==2 and config.field_out in ["EB", "QU"]):
                setattr(alms, attr_name, np.copy(getattr(data, attr_name)))
            elif (getattr(data, attr_name).ndim == 3 and getattr(data, attr_name).shape[1]==2 and config.field_out in ["E", "QU_E"]):
                setattr(alms, attr_name, np.copy(getattr(data, attr_name)[:,0]))
            elif (getattr(data, attr_name).ndim == 3 and getattr(data, attr_name).shape[1]==2 and config.field_out in ["B", "QU_B"]):
                setattr(alms, attr_name, np.copy(getattr(data, attr_name)[:,1]))
    else:
        lmax_in = hp.Alm.getlmax(data.total.shape[-1])
        idx_lmax = np.array([hp.Alm.getidx(lmax_in, ell, m) for ell in range(2, config.lmax + 1) for m in range(ell + 1)])
        idx_config_lmax = np.array([hp.Alm.getidx(config.lmax, ell, m) for ell in range(2, config.lmax + 1) for m in range(ell + 1)])
        for attr_name in vars(data):
            if getattr(data, attr_name).ndim == 2:
                setattr(alms, attr_name, np.zeros((getattr(data, attr_name).shape[0], hp.Alm.getsize(config.lmax)), dtype=complex))
                for i in range(getattr(data, attr_name).shape[0]):
                    getattr(alms, attr_name)[i, idx_config_lmax] = getattr(data, attr_name)[i, idx_lmax]
            if getattr(data, attr_name).ndim == 3:
                if getattr(data, attr_name).shape[1]==3 or (getattr(data, attr_name).shape[1]==2 and config.field_out in ["EB", "QU"]):
                    setattr(alms, attr_name, np.zeros((getattr(data, attr_name).shape[0], getattr(data, attr_name).shape[1], hp.Alm.getsize(config.lmax)), dtype=complex))
                    for i in range(getattr(data, attr_name).shape[0]):
                        getattr(alms, attr_name)[i, :, idx_config_lmax] = getattr(data, attr_name)[i, :, idx_lmax]
                else:
                    setattr(alms, attr_name, np.zeros((getattr(data, attr_name).shape[0], hp.Alm.getsize(config.lmax)), dtype=complex))
                    for i in range(getattr(data, attr_name).shape[0]):
                        if config.field_out in ["E", "QU_E"]:
                            getattr(alms, attr_name)[i, idx_config_lmax] = getattr(data, attr_name)[i, 0, idx_lmax]
                        elif config.field_out in ["B", "QU_B"]:
                            getattr(alms, attr_name)[i, idx_config_lmax] = getattr(data, attr_name)[i, 1, idx_lmax]

    for attr_name in vars(alms):
        for i in range(getattr(alms, attr_name).shape[0]):
            if getattr(alms, attr_name).ndim == 2:
                getattr(alms, attr_name)[i] = hp.almxfl(getattr(alms, attr_name)[i], fell)
            if getattr(alms, attr_name).ndim == 3:
                for j in range(getattr(alms, attr_name).shape[1]):
                    getattr(alms, attr_name)[i, j] = hp.almxfl(getattr(alms, attr_name)[i, j], fell)  

    return alms

def _get_ell_filter(lmin,lmax):
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

def _processing_alms(config: Configs, alms, bring_to_common_resolution=True, pixel_window_in=False):
    if bring_to_common_resolution:
        if config.verbose:
            print("Bringing inputs to common resolution")
        alms = _bring_to_common_resolution(config, alms)
    else:
        if config.verbose:
            print("Inputs are assumed to be at common angular resolution")
    if pixel_window_in:
        if config.verbose:
            print("Correcting for input pixel window function")
        alms = _correct_input_pixwin(config, alms)
    return alms

def _EB_to_QU(EB_maps, lmax, **kwargs):
    alms_ = hp.map2alm(EB_maps, lmax=lmax, pol=False, **kwargs)
    if EB_maps.shape[0] == 3:
        QU_maps = hp.alm2map(alms_,hp.get_nside(EB_maps[0]),lmax=lmax,pol=True)
    else:
        QU_maps = hp.alm2map(np.array([0.*alms_[0],alms_[0],alms_[1]]),hp.get_nside(EB_maps[0]),lmax=lmax,pol=True)[1:]
    return QU_maps

def _E_to_QU(E_map, lmax, **kwargs):
    alms_ = hp.map2alm(E_map, lmax=lmax, pol=False, **kwargs)
    QU_maps = hp.alm2map(np.array([0.*alms_,alms_,0.*alms_]),hp.get_nside(E_map),lmax=lmax,pol=True)[1:]
    return QU_maps

def _B_to_QU(B_map, lmax, **kwargs):
    alms_ = hp.map2alm(B_map, lmax=lmax, pol=False, **kwargs)
    QU_maps = hp.alm2map(np.array([0.*alms_,0.*alms_,alms_]),hp.get_nside(B_map),lmax=lmax,pol=True)[1:]
    return QU_maps

def _bring_to_common_resolution(config: Configs, alms):
    if alms.total.ndim == 2:
        if config.field_out == "T":
            idx_bl = 0
        elif config.field_out in ["E", "QU_E"]:
            idx_bl = 1
        elif config.field_out in ["B", "QU_B"]:
            idx_bl = 2
    
    if config.instrument.beams not in ["gaussian", "file_l", "file_lm"]:
        raise ValueError("Invalid input_beams. It must be either 'gaussian', 'file_l' or 'file_lm'.")

    for i in range(alms.total.shape[0]):
        if config.instrument.beams == "gaussian":
            bl = _bl_from_fwhms(config.fwhm_out,config.instrument.fwhm[i],config.lmax)
        else:
            bl = _bl_from_file(config.instrument.path_beams,config.instrument.channels_tags[i],config.fwhm_out,config.instrument.beams,config.lmax)

        for attr_name in vars(alms):
            if config.instrument.beams != "file_lm":
                if getattr(alms, attr_name).ndim == 2:
                    getattr(alms, attr_name)[i] = hp.almxfl(getattr(alms, attr_name)[i], bl[:,idx_bl])
                if getattr(alms, attr_name).ndim == 3:
                    if getattr(alms, attr_name).shape[1]==3:
                        for j in range(getattr(alms, attr_name).shape[1]):
                            getattr(alms, attr_name)[i, j] = hp.almxfl(getattr(alms, attr_name)[i, j], bl[:,j])
                    if getattr(alms, attr_name).shape[1]==2:
                        for j in range(getattr(alms, attr_name).shape[1]):
                            getattr(alms, attr_name)[i, j] = hp.almxfl(getattr(alms, attr_name)[i, j], bl[:,j+1])    
            else:
                if getattr(alms, attr_name).ndim == 2:
                    getattr(alms, attr_name)[i] = getattr(alms, attr_name)[i] * bl[:,idx_bl]
                if getattr(alms, attr_name).ndim == 3:
                    if getattr(alms, attr_name).shape[1]==3:
                        for j in range(getattr(alms, attr_name).shape[1]):
                            getattr(alms, attr_name)[i, j] = getattr(alms, attr_name)[i, j] * bl[:,j]
                    if getattr(alms, attr_name).shape[1]==2:
                        for j in range(getattr(alms, attr_name).shape[1]):
                            getattr(alms, attr_name)[i, j] = getattr(alms, attr_name)[i, j] * bl[:,j+1]

    return alms

def _bl_from_fwhms(fwhm_out,fwhm_in,lmax):
    bl_in = hp.gauss_beam(np.radians(fwhm_in/60.), lmax=lmax,pol=True)
    bl_out = hp.gauss_beam(np.radians(fwhm_out/60.), lmax=lmax,pol=True)
    return bl_out / bl_in

def _bl_from_file(beam_path,channel,fwhm_out,input_beams,lmax):
    beam_file = beam_path + f'/beam_TEB_{channel}.fits'
    
    bl_in = _get_beam_from_file(beam_file,lmax,symmetric_beam=False if input_beams == "file_lm" else True)
    bl_out_l = hp.gauss_beam(np.radians(fwhm_out/60.), lmax=lmax,pol=True)

    if input_beams == "file_l":
        return (bl_out_l[:,:3]) / (bl_in[:,:3])
    elif input_beams == "file_lm":
        bl_out = np.zeros((hp.Alm.getsize(lmax),3),dtype=complex)
        for ell in range(0, lmax + 1):
            idx_lmax = np.array([hp.Alm.getidx(lmax, ell, m) for m in range(ell + 1)])
            bl_out[idx_lmax,:] = np.tile(bl_out_l[ell,:3], (ell+1, 1))
        return bl_out / (bl_in[:,:3]) 

def _get_beam_from_file(beam_file,lmax,symmetric_beam=True):
    hdul = fits.open(beam_file)
    primary_hdu = hdul[1].data
    bl_file = np.column_stack([primary_hdu[col].astype(str).astype(float) for col in primary_hdu.names]).squeeze()

    if bl_file.ndim != 2 or (bl_file.shape[1] != 3 and bl_file.shape[1] != 4):
        raise ValueError("Beam file must be 2-dimensional and have 3 or 4 columns (for T, E, B and EB).")

    if symmetric_beam:
        if lmax > (bl_file.shape[0]-1):
            raise ValueError("Beam file does not have enough multipoles.")
        return bl_file[:lmax+1]
    else:
        if lmax > hp.Alm.getlmax(bl_file.shape[0]):
            raise ValueError("The provided asymmetric beam file does not have enough values for the given alm.")
        bl = np.zeros((hp.Alm.getsize(lmax),bl_file.shape[1]),dtype=complex)
        lmax_file = hp.Alm.getlmax(bl_file.shape[0])
        for ell in range(0, lmax + 1):
            idx_lmax = np.array([hp.Alm.getidx(lmax, ell, m) for m in range(ell + 1)])
            idx_lmax_file = np.array([hp.Alm.getidx(lmax_file, ell, m) for m in range(ell + 1)])
            bl[idx_lmax,:] = bl_file[idx_lmax_file,:]
        return bl


def _correct_input_pixwin(config: Configs, alms):
    pixwin_in = hp.pixwin(config.nside_in, pol=True, lmax=config.lmax)
    for attr_name in vars(alms):
        if getattr(alms, attr_name).ndim == 2:
            pw = pixwin_in[0] if config.field_out == "T" else pixwin_in[1]
            for i in range(getattr(alms, attr_name).shape[0]):
                getattr(alms, attr_name)[i] = hp.almxfl(getattr(alms, attr_name)[i], pw)
        elif getattr(alms, attr_name).ndim == 3:
            if getattr(alms, attr_name).shape[1]==2:
                pw = pixwin_in[1]
                for i, j in np.ndindex(getattr(alms, attr_name).shape[0],getattr(alms, attr_name).shape[1]):
                    getattr(alms, attr_name)[i, j] = hp.almxfl(getattr(alms, attr_name)[i, j], pw)
            if getattr(alms, attr_name).shape[1]==3:
                for i, j in np.ndindex(getattr(alms, attr_name).shape[0],getattr(alms, attr_name).shape[1]):
                    getattr(alms, attr_name)[i, j] = hp.almxfl(getattr(alms, attr_name)[i, j], pixwin_in[0]) if j==0 else hp.almxfl(getattr(alms, attr_name)[i, j], pixwin_in[1])
    return alms

def _get_needlet_windows_(needlet_config, lmax):
    needlet_options = {
    "cosine": get_cosine_windows,
    'standard': get_standard_windows,
    'mexican': get_mexican_windows}

    b_ell = needlet_options[needlet_config["needlet_windows"]](lmax, needlet_config)
    return b_ell
    
def get_cosine_windows(lmax, needlet_config):
    if not "ell_peaks" in needlet_config:
        raise ValueError("Needlet window type 'cosine' requires 'ell_peaks' list. \
            ell_peaks is a list of integers that define the multipoles of the peaks of the cosine windows.")
    def compute_cosine_window(ell_peak, ell_ext):
        ell_range = np.arange(np.min([ell_peak, ell_ext]), np.max([ell_peak, ell_ext]) + 1)
        return np.cos((ell_peak - ell_range) / (ell_peak - ell_ext) * np.pi / 2.)

    bandpeaks = needlet_config["ell_peaks"]
    b_ell = np.zeros((len(bandpeaks),lmax+1))
    if bandpeaks[0] > 0:
        b_ell[0,:bandpeaks[0]]=1.
    b_ell[0,bandpeaks[0]:bandpeaks[1]+1] = compute_cosine_window(bandpeaks[0], bandpeaks[1]) #
    if len(bandpeaks) >= 3:
        for i in range(1, len(bandpeaks)-1):
            ell_min=bandpeaks[i-1]
            ell_p=bandpeaks[i]
            ell_max=bandpeaks[i+1]
            b_ell[i,ell_min:ell_p+1] = compute_cosine_window(ell_peak=ell_p, ell_ext=ell_min)
            b_ell[i,ell_p:ell_max+1] = compute_cosine_window(ell_peak=ell_p, ell_ext=ell_max)
    b_ell[-1,bandpeaks[-2]:bandpeaks[-1]+1] = compute_cosine_window(ell_peak=bandpeaks[-1], ell_ext=bandpeaks[-2])
    if bandpeaks[-1] < lmax:
        b_ell[-1,bandpeaks[-1]+1:] = 1.
    return b_ell

def get_standard_windows(lmax, needlet_config):
    if not "width" in needlet_config:
        raise ValueError("'standard' needlet windows require 'width' parameter. \
            'width' is a floating number that defines the width of the standard needlet windows.")
    j_min = 0
    j_max = 2
    b_ell = nl.standardneedlet(needlet_config["width"],list(range(j_min,j_max+1)),lmax)
    while np.abs(np.sum(b_ell**2,axis=0)[-1]-1.) > 1.e-5:
        j_max += 1
        b_ell = nl.standardneedlet(needlet_config["width"],list(range(j_min,j_max+1)),lmax)
    
    if not "merging_needlets" in needlet_config:
        return b_ell
    else:
        return _merge_needlets_(b_ell, needlet_config["merging_needlets"])

def get_mexican_windows(lmax, needlet_config):
    if not "width" in needlet_config:
        raise ValueError("'mexican' needlet windows require 'width' parameter. \
            'width' is a floating number that defines the width of the mexican needlet windows.")
    j_min = 0
    j_max = 2
    b_ell = nl.mexicanneedlet(needlet_config["width"],list(range(j_min,j_max+1)),lmax)
    while np.abs(np.sum(b_ell**2,axis=0)[-1]-1.) > 1e-5:
        j_max += 1
        b_ell = nl.mexicanneedlet(needlet_config["width"],list(range(j_min,j_max+1)),lmax)
    
    if not "merging_needlets" in needlet_config:
        return b_ell
    else:
        return _merge_needlets_(b_ell, needlet_config["merging_needlets"])
    
def _merge_needlets_(b_ell, merging_needlets):
    if isinstance(merging_needlets, int):
        if b_ell.shape[0] <= merging_needlets:
            raise Warning("Number of needlets to merge in the first band is larger than the number of needlets. \
                All needlets are merged in one band. You are therefore running in pixel domain.")
            return np.sqrt(np.sum(b_ell**2, axis=0)).reshape(1,b_ell.shape[1])
        else:
            return np.concatenate([np.sqrt(np.sum((b_ell**2)[:merging_needlets],axis=0)).reshape(1,b_ell.shape[1]), b_ell[merging_needlets:]])
    elif isinstance(merging_needlets, list):
        if b_ell.shape[0] <= merging_needlets[0]:
            raise Warning("Number of needlets to merge in the first band is larger than the number of needlets. \
                All needlets are therefore merged in one band and the method will be performed in pixel domain.")
            return  np.sqrt(np.sum(b_ell**2, axis=0)).reshape(1,b_ell.shape[1])
        elif b_ell.shape[0] > merging_needlets[-1]:
            if merging_needlets[0] > 0:
                merging_needlets.insert(0,0)
            merged_b_ell = []
            for j_low, j_high in zip(merging_needlets[:-1], merging_needlets[1:]):
                merged_b_ell.append(np.sqrt(np.sum((b_ell**2)[j_low:j_high],axis=0)))
            merged_b_ell = np.concatenate([np.array(merged_b_ell),b_ell[merging_needlets[-1]:]])
            return merged_b_ell
        elif b_ell.shape[0] <= merging_needlets[-1]:
            if merging_needlets[0] > 0:
                merging_needlets.insert(0,0)
            merging_needlets = np.array(merging_needlets)
            merging_needlets = merging_needlets[merging_needlets < b_ell.shape[0]]
            merging_needlets = np.append(merging_needlets, b_ell.shape[0])
            merged_b_ell = []
            for j_low, j_high in zip(merging_needlets[:-1], merging_needlets[1:]):
                merged_b_ell.append(np.sqrt(np.sum((b_ell**2)[j_low:j_high],axis=0)))
            return np.array(merged_b_ell)
    else:
        raise ValueError("merging_needlets must be an integer or a list of integers")

def _get_nside_lmax_from_b_ell(b_ell,nside,lmax):
    max_b = np.max(np.nonzero(b_ell)) / 2
    nside_values = [(2**(k+1), 2**(k)) for k in range(3, 12)]

    nside_ = None
    if max_b <= 8:
        nside_ = 8
    else:
        for upper_bound, lower_bound in nside_values:
            if lower_bound < max_b <= upper_bound:
                nside_ = upper_bound
    if nside_:    
        lmax_ = 2 * nside_
    else:
        nside_ = nside
        lmax_ = lmax
    return nside_, lmax_

def _needlet_filtering(alms, b_ell, lmax_out):
    if alms.ndim == 2:
        filtered_alms = np.array([hp.almxfl(alms[:,c], b_ell) for c in range(alms.shape[-1])]).T
    elif alms.ndim == 1:
        filtered_alms = hp.almxfl(alms, b_ell)

    if alms.shape[0] == hp.Alm.getsize(lmax_out):
        return filtered_alms
    else:
        lmax_j = np.min([hp.Alm.getlmax(alms.shape[0]), lmax_out])
        idx_lmax_in = np.array([hp.Alm.getidx(hp.Alm.getlmax(filtered_alms.shape[0]), ell, m) for ell in range(0, lmax_j+1) for m in range(ell + 1)])
        idx_lmax_out = np.array([hp.Alm.getidx(lmax_out, ell, m) for ell in range(0, lmax_j+1) for m in range(ell + 1)])
        if alms.ndim == 2:
            alms_j = np.zeros((hp.Alm.getsize(lmax_out), alms.shape[-1]), dtype=complex)
            alms_j[idx_lmax_out, :] = filtered_alms[idx_lmax_in, :]
        elif alms.ndim == 1:
            alms_j = np.zeros((hp.Alm.getsize(lmax_out)), dtype=complex)
            alms_j[idx_lmax_out] = filtered_alms[idx_lmax_in]
        return alms_j

def _get_local_cov(input_maps, lmax, ilc_bias, b_ell=None, mask=None, reduce_bias=False, input_maps_2=None):
    if not isinstance(b_ell, np.ndarray):
        b_ell = np.ones(lmax+1)
    nmodes_band  = np.sum((2.*np.arange(0,lmax+1)+1.) * (b_ell)**2 )
    pps = np.sqrt(float(input_maps.shape[1]) * float(input_maps.shape[0]-1) / (ilc_bias * nmodes_band) )

    if mask is None:
        cov = np.zeros((input_maps.shape[0],input_maps.shape[0],int(np.min([input_maps.shape[1],12*128**2]))))
    else:
        cov = np.zeros((input_maps.shape[0],input_maps.shape[0],input_maps.shape[1]))

    for i in range(input_maps.shape[0]):
        for k in range(i,input_maps.shape[0]):
            if input_maps_2 is None:
                if mask is None:
                    cov[i,k]=_get_local_cov_(input_maps[i], input_maps[k], pps, int(np.min([hp.npix2nside(input_maps.shape[1]),128])), reduce_bias=reduce_bias)
                else:
                    cov[i,k]=_get_local_cov_((input_maps[i]) * mask, (input_maps[k]) * mask, pps, int(hp.npix2nside(input_maps.shape[1])), reduce_bias=reduce_bias)
            else:
                if mask is None:
                    cov[i,k]=_get_local_cov_(input_maps[i], input_maps_2[k], pps, int(np.min([hp.npix2nside(input_maps.shape[1]),128])), reduce_bias=reduce_bias)
                else:
                    cov[i,k]=_get_local_cov_((input_maps[i]) * mask, (input_maps_2[k]) * mask, pps, int(hp.npix2nside(input_maps.shape[1])), reduce_bias=reduce_bias)

    for i in range(input_maps.shape[0]):
        for k in range(i):
            cov[i,k]=cov[k,i]
            
    return cov

def _get_local_cov_(map1, map2, pps, *nside_covar, reduce_bias=False):

    map_ = map1 * map2
    npix_ = map_.size
    nside_ = hp.get_nside(map_)
    if not nside_covar:
        nside_covar = nside_
    else:
        nside_covar = nside_covar[0]

    # First degrade a bit to speed-up smoothing

    if (nside_ / 4) > 1:
        nside_out = int(nside_ / 4)
    else:
        nside_out = 1

    stat = hp.ud_grade(map_, nside_out = nside_out, order_in = 'RING', order_out = 'RING')
    
    # Compute alm

    lmax_stat = 2 * nside_out #3 * nside_out - 1 # 
    alm_s = hp.map2alm(stat, lmax=lmax_stat, iter=1, use_weights=True)# iter=0?

    # Find smoothing size

    pix_size = np.sqrt(4.0 * np.pi / npix_)
    fwhm_stat = pps * pix_size
    bl_stat = hp.sphtfunc.gauss_beam(fwhm_stat, lmax_stat)
    if reduce_bias:
        thetas = np.arange(0,np.pi,0.002)
        beam_stat = hp.bl2beam(bl_stat, thetas)
        theta_ = 0.5 * fwhm_stat
        dist = 0.5 * (np.tanh(30*(thetas-0.3*theta_)) - np.tanh(30*(thetas-3*theta_)))
        dist /= np.max(dist)
        dist[np.argmax(dist):]=1.
        bl_stat = hp.beam2bl(dist*beam_stat,thetas,lmax=lmax_stat)

    alm_s = hp.sphtfunc.almxfl(alm_s, bl_stat)
    
    # Back to pixel space

    stat_out = hp.alm2map(alm_s, nside_covar, lmax=lmax_stat)

    return stat_out   


def _get_local_cov_new_(map1, map2, pps, *nside_covar):

    map_ = map1 * map2
    npix_ = map_.size
    nside_ = hp.get_nside(map_)
    if not nside_covar:
        if (nside_ / 4) > 1:
            nside_covar = int(nside_ / 4)
        else:
            nside_covar = 1
    else:
        nside_covar = nside_covar[0]

    # Compute alm

    lmax_stat = 2 * nside_covar #3 * nside_out - 1 # 
    alm_s = hp.sphtfunc.map2alm(map_, lmax=lmax_stat, iter=1, use_weights=True)# iter=0?

    # Find smoothing size

    pix_size = np.sqrt(4.0 * np.pi / npix_)
    fwhm_stat = pps * pix_size
    bl_stat = hp.sphtfunc.gauss_beam(fwhm_stat, lmax_stat)

    alm_s = hp.sphtfunc.almxfl(alm_s, bl_stat)
    
    # Back to pixel space

    stat_out = hp.alm2map(alm_s, nside_covar, lmax=lmax_stat)

    return stat_out    

def _save_compsep_products(config: Configs, output_maps, compsep_run, nsim=None):
#    if config_dir_outputs[-1] != "/":
#        config_dir_outputs += "/"
    path_out = _get_full_path_out(config, compsep_run)

    for attr_name, attr_values in vars(output_maps).items():
        if attr_name == "total":
            label_out = "output_total"
        elif attr_name == "cmb":
            label_out = "output_cmb"
        elif attr_name == "m":
            label_out = "fgd_complexity"
        else:
            label_out = attr_name + "_residuals"

        path_c = os.path.join(path_out, f"{label_out}")
        os.makedirs(path_c, exist_ok=True)
        if compsep_run["method"] in ["gilc","gpilc"]:
            for f, freq in enumerate(compsep_run["channels_out"]):
                if nsim is None:
                    hp.write_map(f"{path_c}/{config.field_out}_{label_out}_{config.instrument.channels_tags[freq]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}.fits", attr_values[f], overwrite=True)
                else:
                    hp.write_map(f"{path_c}/{config.field_out}_{label_out}_{config.instrument.channels_tags[freq]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}_{nsim}.fits", attr_values[f], overwrite=True)
        elif (compsep_run["method"] in ["fgd_diagnostic", "fgd_P_diagnostic"]) and (compsep_run["domain"]=="needlet"):
            for j in range(attr_values.shape[0]):
                filename = f"{path_c}/{config.field_out}_{label_out}_nl{j}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
                if nsim is not None:
                    filename += f"_{nsim}"
                filename += ".fits"
                if attr_values.ndim == 2:
                    hp.write_map(filename, attr_values[j], overwrite=True)
                elif attr_values.ndim == 3:
                    hp.write_map(filename, attr_values[:,j], overwrite=True)
        else:
            if nsim is None:
                hp.write_map(f"{path_c}/{config.field_out}_{label_out}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}.fits", attr_values, overwrite=True)
            else:
                hp.write_map(f"{path_c}/{config.field_out}_{label_out}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}_{nsim}.fits", attr_values, overwrite=True)

def _save_residuals_template(config: Configs, output_maps, compsep_run, nsim=None):
#    if config_dir_outputs[-1] != "/":
#        config_dir_outputs += "/"
    path_out = compsep_run["compsep_path"]

    gnilc_run = (re.search(r'(gilc_[^/]+)', compsep_run["gnilc_path"])).group(1)
    if "needlet" in gnilc_run:
        folder_after = (compsep_run["gnilc_path"]).split(gnilc_run + "/")[1].split("/")[0]
        gnilc_run += f"_{folder_after}"

    for attr_name, attr_values in vars(output_maps).items():
        if attr_name == "total":
            label_out = "fgres_templates"
        elif attr_name == "fgds":
            label_out = "fgres_templates_ideal"
        else:
            label_out = "fgres_templates_" + attr_name

        path_c = os.path.join(path_out, f"{label_out}", gnilc_run)
        os.makedirs(path_c, exist_ok=True)
        if nsim is None:
            hp.write_map(f"{path_c}/{config.field_out}_{label_out}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}.fits", attr_values, overwrite=True)
        else:
            hp.write_map(f"{path_c}/{config.field_out}_{label_out}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}_{nsim}.fits", attr_values, overwrite=True)

def _get_full_path_out(config: Configs, compsep_run):
    if compsep_run["method"] in ["mc_ilc", "c_ilc", "c_pilc"]:
        complete_path = f'{compsep_run["method"]}_{compsep_run["domain"]}_bias{compsep_run["ilc_bias"]}_nls{"-".join([str(x) for x in compsep_run["special_nls"]])}' 
    elif compsep_run["method"] == "mcilc":
        complete_path = f'{compsep_run["method"]}_{compsep_run["domain"]}'
    elif compsep_run["method"] in ["gilc", "gpilc"]:
        complete_path = f'{compsep_run["method"]}_{compsep_run["domain"]}_bias{compsep_run["ilc_bias"]}'
        if compsep_run["domain"] == "pixel":
            if compsep_run["m_bias"] != 0:
                complete_path += f"_m+{compsep_run['m_bias']}" if compsep_run["m_bias"] > 0 else f"_m{compsep_run['m_bias']}"
            if compsep_run["depro_cmb"] is not None:
                complete_path += f"_deproCMB{compsep_run['depro_cmb']}"
        elif compsep_run["domain"] == "needlet":
            if any(np.array(compsep_run["m_bias"]) != 0):
                unique_m_bias = np.unique(np.array(compsep_run["m_bias"])[np.array(compsep_run["m_bias"]) != 0])
                for m_bias in unique_m_bias:
                    nls_bias = np.where(np.array(compsep_run["m_bias"]) == m_bias)[0]
                    if m_bias > 0:
                        complete_path += f"_m+{m_bias}_nls{'-'.join([str(x) for x in nls_bias])}"
                    else:
                        complete_path += f"_m{m_bias}_nls{'-'.join([str(x) for x in nls_bias])}"
            if any(np.array(compsep_run["depro_cmb"]) != None):
                unique_depro_cmb = np.unique(np.array(compsep_run["depro_cmb"])[np.array(compsep_run["depro_cmb"]) != None])
                for depro_cmb in unique_depro_cmb:
                    nls_depro = np.where(np.array(compsep_run["depro_cmb"]) == depro_cmb)[0]
                    complete_path += f"_deproCMB{depro_cmb}_nls{'-'.join([str(x) for x in nls_depro])}"
    else:
        complete_path = f'{compsep_run["method"]}_{compsep_run["domain"]}_bias{compsep_run["ilc_bias"]}'
    
    if config.leakage_correction is not None:
        leak_def = (config.leakage_correction).split("_")[0] + (config.leakage_correction).split("_")[1] 
        if "_recycling" in config.leakage_correction:
            if "_iterations" in config.leakage_correction:
                iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
                leak_def += f'_iters{iterations}'
        complete_path += f"_{leak_def}"

    if compsep_run["method"] in ["cilc", "c_ilc","cpilc", "c_pilc"]:
        if compsep_run["domain"] == "pixel":
            mom_text = "".join(compsep_run["constraints"]["moments"])
        elif compsep_run["domain"] == "needlet":
            if all(list(set(row)) == list(set(compsep_run["constraints"]["moments"][0])) for row in compsep_run["constraints"]["moments"]):
                mom_text = "".join(compsep_run["constraints"]["moments"][0])
            else:
                mom_text = ""
                for idx, row in enumerate(compsep_run["constraints"]["moments"]):
                    if idx == 0:
                        mom_text += "".join(row)
                    else:
                        if list(set(row)) != list(set(compsep_run["constraints"]["moments"][idx-1])):
                            mom_text += "_" + "".join(row)
        if compsep_run["domain"] == "pixel":
            all_depros = list(set(compsep_run["constraints"]["deprojection"]))
        elif compsep_run["domain"] == "needlet":
            #dim_depro = len(compsep_run["constraints"]["deprojection"]) if isinstance(compsep_run["constraints"]["deprojection"][0], list) else 1
            #if dim_depro ==1:
            #    all_depros = list(set(compsep_run["constraints"]["deprojection"]))
            #else:
            all_depros = list(set(element for sublist in compsep_run["constraints"]["deprojection"] for element in sublist))
        if len(all_depros)==1:
            if all_depros[0] != 0.:
                mom_text += f"_depro{all_depros[0]}"
        else:
            mom_text += "_mixeddepro"
        complete_path = os.path.join(complete_path, mom_text)

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

    if compsep_run["method"] in ["mcilc","mc_ilc"]:
        text_ = compsep_run["mc_type"]
        for freq_tracer in compsep_run["channels_tracers"]:
            text_ += f"_{config.instrument.channels_tags[freq_tracer]}"
        complete_path = os.path.join(complete_path, text_)

    path_out = os.path.join(config.path_outputs, complete_path)

    return path_out

def merge_dicts(d):
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
          
def obj_to_array(obj):
    """
    Convert an object with attributes to a numpy array.
    """
    if isinstance(obj, SimpleNamespace):
        allowed_attributes = ["total", "fgds", "noise", "cmb", "dust", "synch", "ame", "co", "freefree", "cib", "tsz", "ksz", "radio_galaxies"]
        array = []
        for attr in allowed_attributes:
            if hasattr(obj, attr):
                array.append(getattr(obj, attr))
        array = np.array(array)
        if array.ndim == 3:
            return np.transpose(array, axes=(1,2,0))
        elif array.ndim == 4:
            return np.transpose(array, axes=(1,2,3,0))
    else:
        raise ValueError("Input must be a SimpleNamespace object.")

def obj_out_to_array(obj):
    """
    Convert an object with attributes to a numpy array.
    """
    if isinstance(obj, SimpleNamespace):
        allowed_attributes = ["output_total", "noise_residuals", "fgds_residuals", "output_cmb", "fgres_templates", "fgres_templates_noise", "fgres_templates_ideal"]
        array = []
        for attr in allowed_attributes:
            if hasattr(obj, attr):
                array.append(getattr(obj, attr))
        return np.array(array)
    else:
        raise ValueError("Input must be a SimpleNamespace object.")

def array_to_obj(array, obj):
    """
    Convert a numpy array to an object with attributes.
    """
    allowed_attributes = ["total", "fgds", "noise", "cmb", "dust", "synch", "ame", "co", "freefree", "cib", "tsz", "ksz", "radio_galaxies"]
    new_obj = SimpleNamespace()

    count = 0
    for attr in allowed_attributes:
        if hasattr(obj, attr):
            setattr(new_obj, attr, array[..., count])
            count += 1
    return new_obj

def _slice_data(data, field_in, field_out):
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

def _slice_outputs(outputs, field_in, field_out):
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

def _slice_data_for_cls(data, field_in, field_out):
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

def _map2alm_kwargs(**kwargs):
    """
    Helper function to create a dictionary of keyword arguments for hp.map2alm.
    """
    map2alm_kwargs = {}
    # Define the allowed keywords for each function
    allowed_map2alm_kwargs = {"iter", "mmax", "use_weights", "datapath", "use_pixel_weights"}
    # Distribute kwargs
    for key, value in kwargs.items():
        if key in allowed_map2alm_kwargs:
            map2alm_kwargs[key] = value
    return map2alm_kwargs