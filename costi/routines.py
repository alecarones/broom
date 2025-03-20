import numpy as np
import healpy as hp
import mtneedlet as nl
from astropy.io import fits
import re
import os
from .configurations import Configs
from .leakage import purify_master, purify_recycling

def _alms_from_data(config: Configs, data, mask_in=None):
    fell = _get_ell_filter(2,config.lmax)

    if config.data_type == "maps":
        alms = _maps_to_alms(config, data, mask_in=mask_in)
    elif config.data_type == "alms":
        alms = _alms_to_alms(config, data)

    alms = _processing_alms(config, alms)    

    return alms

def _maps_to_alms(config: Configs, data, mask_in=None):
    fell = _get_ell_filter(2,config.lmax)

    if mask_in is None:
        mask_in = np.ones(data.shape[-2])
    elif not isinstance(mask_in, np.ndarray): 
        raise ValueError("Invalid mask. It must be a numpy array.")
    elif hp.get_nside(mask_in) != hp.npix2nside(data.shape[-2]):
            raise ValueError("Mask HEALPix resolution does not match data HEALPix resolution.")
    mask_in /= np.max(mask_in)

    if config.leakage_correction is not None:
        if "_recycling" in config.leakage_correction:
            if "_iterations" in config.leakage_correction:
                iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
            else:
                iterations = 0

    if data.ndim == 3:
        alms = np.zeros((data.shape[0], hp.Alm.getsize(config.lmax), data.shape[-1]), dtype=complex)
        for i, c in np.ndindex(data.shape[0],data.shape[-1]):
            alms[i, :, c] = hp.almxfl(hp.map2alm((data[i, :, c] * mask_in), lmax=config.lmax, pol=False),fell)
    else:
        if data.shape[1]==3:
            alms = np.zeros((data.shape[0], data.shape[1], hp.Alm.getsize(config.lmax), data.shape[-1]), dtype=complex)
            for i, c in np.ndindex(data.shape[0],data.shape[-1]):
                if (config.leakage_correction is None) or (config.field_in=="TEB") or (np.mean(mask_in**2)==1.):
                    alms[i, :, :, c] = hp.map2alm((data[i, :, :, c] * mask_in), lmax=config.lmax, pol=config.field_in=="TQU")
                else:
                    alms[i, 0, :, c] = hp.map2alm((data[i, 0, :, c] * mask_in), lmax=config.lmax)
                    if "_purify" in config.leakage_correction:
                        alms[i, 1:, :, c] = purify_master(data[i, 1:, :, c], mask_in, config.lmax,purify_E=("E" in config.leakage_correction))
                    elif "_recycling" in config.leakage_correction:
                        if c==0:
                            alms[i, 1:] = purify_recycling(data[i, 1:], mask_in, config.lmax,purify_E=("E" in config.leakage_correction), iterations=iterations)
                for j in range(alms.shape[1]):
                    alms[i, j, :, c] = hp.almxfl(alms[i, j, :, c], fell)

        elif data.shape[1]==2:
            if config.field_out in ["E", "B", "QU_E", "QU_B"]:
                alms = np.zeros((data.shape[0], hp.Alm.getsize(config.lmax), data.shape[-1]), dtype=complex)
            else:
                alms = np.zeros((data.shape[0], data.shape[1], hp.Alm.getsize(config.lmax), data.shape[-1]), dtype=complex)
            for i, c in np.ndindex(data.shape[0],data.shape[-1]):
                if (config.leakage_correction is None) or (np.mean(mask_in**2)==1.) or (config.field_in=="EB") or (config.field_out in ["E", "QU_E"] and ("E" not in config.leakage_correction)) or (config.field_out in ["B", "QU_B"] and ("B" not in config.leakage_correction)):
                    alms[i, ..., c] = hp.map2alm(np.vstack((0. * data[i, 0, :, c], data[i, :, :, c])) * mask_in, lmax=config.lmax, pol=config.field_in=="QU")[[1,2] if config.field_out in ["QU", "EB"] else 1 if config.field_out in ["E", "QU_E"] else 2]
                else:
                    if "_purify" in config.leakage_correction:
                        alms_pure = purify_master(data[i, ..., c], mask_in, config.lmax,purify_E=("E" in config.leakage_correction))
                        alms[i, ..., c] = np.copy(alms_pure) if config.field_out in ["QU", "EB"] else np.copy(alms_pure[0]) if config.field_out in ["E", "QU_E"] else np.copy(alms_pure[1])
                    elif "_recycling" in config.leakage_correction:
                        if c==0:
                            alms_pure = purify_recycling(data[i], mask_in, config.lmax,purify_E=("E" in config.leakage_correction), iterations=iterations)
                            alms[i] = np.copy(alms_pure) if config.field_out in ["QU", "EB"] else np.copy(alms_pure[0]) if config.field_out in ["E", "QU_E"] else np.copy(alms_pure[1])
                if config.field_out in ["E", "B", "QU_E", "QU_B"]:
                    alms[i, :, c] = hp.almxfl(alms[i, :, c], fell)
                else:
                    for j in range(alms.shape[1]):
                        alms[i, j, :, c] = hp.almxfl(alms[i, j, :, c], fell)
    return alms

def _alms_to_alms(config: Configs, data):
    fell = _get_ell_filter(2,config.lmax)

    if hp.Alm.getlmax(data.shape[-2]) == config.lmax:
        if data.ndim == 3 or (data.ndim == 4 and data.shape[1]==3) or (data.ndim == 4 and data.shape[1]==2 and config.field_out in ["EB", "QU"]):
            alms = np.copy(data)
        elif (data.ndim == 4 and data.shape[1]==2 and config.field_out in ["E", "QU_E"]):
            alms = np.copy(data[:,0])
        elif (data.ndim == 4 and data.shape[1]==2 and config.field_out in ["B", "QU_B"]):
            alms = np.copy(data[:,1])
    else:
        lmax_in = hp.Alm.getlmax(data.shape[-2])
        idx_lmax = np.array([hp.Alm.getidx(lmax_in, ell, m) for ell in range(2, config.lmax + 1) for m in range(ell + 1)])
        idx_config_lmax = np.array([hp.Alm.getidx(config.lmax, ell, m) for ell in range(2, config.lmax + 1) for m in range(ell + 1)])
        if data.ndim == 3:
            alms = np.zeros((data.shape[0], hp.Alm.getsize(config.lmax), data.shape[-1]), dtype=complex)
            for i, c in np.ndindex(data.shape[0],data.shape[-1]):
                alms[i, idx_config_lmax, c] = data[i, idx_lmax, c]
        if data.ndim == 4:
            if data.shape[1]==3 or (data.shape[1]==2 and config.field_out in ["EB", "QU"]):
                alms = np.zeros((data.shape[0], data.shape[1], hp.Alm.getsize(config.lmax), data.shape[-1]), dtype=complex)
                for i, c in np.ndindex(data.shape[0],data.shape[-1]):
                    alms[i, :, idx_config_lmax, c] = data[i, :, idx_lmax, c]
            else:
                alms = np.zeros((data.shape[0], hp.Alm.getsize(config.lmax), data.shape[-1]), dtype=complex)
                for i, c in np.ndindex(data.shape[0],data.shape[-1]):
                    if config.field_out in ["E", "QU_E"]:
                        alms[i, idx_config_lmax, c] = data[i, 0, idx_lmax, c]
                    elif config.field_out in ["B", "QU_B"]:
                        alms[i, idx_config_lmax, c] = data[i, 1, idx_lmax, c]

    for i, c in np.ndindex(alms.shape[0],alms.shape[-1]):
        if alms.ndim == 3:
            alms[i, :, c] = hp.almxfl(alms[i, :, c], fell)
        if alms.ndim == 4:
            for j in range(alms.shape[1]):
                alms[i, j, :, c] = hp.almxfl(alms[i, j, :, c], fell)  
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

def _processing_alms(config: Configs, alms):
    if config.bring_to_common_resolution:
        if config.verbose:
            print("Bringing inputs to common resolution")
        alms = _bring_to_common_resolution(config, alms)
    else:
        if config.verbose:
            print("Inputs are assumed to be at common angular resolution")
    if (config.pixel_window_in) and (config.data_type == "maps"):
        if config.verbose:
            print("Correcting for input pixel window function")
        alms = _correct_input_pixwin(config, alms)
    return alms

def _EB_to_QU(EB_maps, lmax):
    alms_ = hp.map2alm(EB_maps, lmax=lmax, pol=False)
    if EB_maps.shape[0] == 3:
        QU_maps = hp.alm2map(alms_,hp.get_nside(EB_maps[0]),lmax=lmax,pol=True)
    else:
        QU_maps = hp.alm2map(np.array([0.*alms_[0],alms_[0],alms_[1]]),hp.get_nside(EB_maps[0]),lmax=lmax,pol=True)[1:]
    return QU_maps

def _E_to_QU(E_map, lmax):
    alms_ = hp.map2alm(E_map, lmax=lmax, pol=False)
    QU_maps = hp.alm2map(np.array([0.*alms_,alms_,0.*alms_]),hp.get_nside(E_map),lmax=lmax,pol=True)[1:]
    return QU_maps

def _B_to_QU(B_map, lmax):
    alms_ = hp.map2alm(B_map, lmax=lmax, pol=False)
    QU_maps = hp.alm2map(np.array([0.*alms_,0.*alms_,alms_]),hp.get_nside(B_map),lmax=lmax,pol=True)[1:]
    return QU_maps

def _bring_to_common_resolution(config: Configs, alms):
    if alms.ndim == 3:
        if config.field_out == "T":
            idx_bl = 0
        elif config.field_out in ["E", "QU_E"]:
            idx_bl = 1
        elif config.field_out in ["B", "QU_B"]:
            idx_bl = 2
    
    if config.input_beams not in ["gaussian", "file_l", "file_lm"]:
        raise ValueError("Invalid input_beams. It must be either 'gaussian', 'file_l' or 'file_lm'.")

    for i in range(alms.shape[0]):
        if config.input_beams == "gaussian":
            bl = _bl_from_fwhms(config.fwhm_out,config.instrument.fwhm[i],config.lmax)
        else:
            bl = _bl_from_file(config.beams_path,config.instrument.channels_tags[i],config.fwhm_out,config.input_beams,config.lmax)

        for c in range(alms.shape[-1]):
            if config.input_beams != "file_lm":
                if alms.ndim == 3:
                    alms[i, :, c] = hp.almxfl(alms[i, :, c], bl[:,idx_bl])
                if alms.ndim == 4:
                    if alms.shape[1]==3:
                        for j in range(alms.shape[1]):
                            alms[i, j, :, c] = hp.almxfl(alms[i, j, :, c], bl[:,j])
                    if alms.shape[1]==2:
                        for j in range(alms.shape[1]):
                            alms[i, j, :, c] = hp.almxfl(alms[i, j, :, c], bl[:,j+1])    
            else:
                if alms.ndim == 3:
                    alms[i, :, c] = alms[i, :, c] * bl[:,idx_bl]
                if alms.ndim == 4:
                    if alms.shape[1]==3:
                        for j in range(alms.shape[1]):
                            alms[i, j, :, c] = alms[i, j, :, c] * bl[:,j]
                    if alms.shape[1]==2:
                        for j in range(alms.shape[1]):
                            alms[i, j, :, c] = alms[i, j, :, c] * bl[:,j+1]

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

    if bl_file.ndim != 2 or bl_file.shape[1] != 3 or bl_file.shape[1] != 4:
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
    if alms.ndim == 3:
        pw = pixwin_in[0] if config.field_out == "T" else pixwin_in[1]
        for i, c in np.ndindex(alms.shape[0],alms.shape[-1]):
            alms[i, :, c] = hp.almxfl(alms[i, :, c], pw)
    elif alms.ndim == 4:
        if alms.shape[1]==2:
            pw = pixwin_in[1]
            for i, c, j in np.ndindex(alms.shape[0],alms.shape[-1],alms.shape[1]):
                alms[i, j, :, c] = hp.almxfl(alms[i, j, :, c], pw)
        if alms.shape[1]==3:
            for i, c, j in np.ndindex(alms.shape[0],alms.shape[-1],alms.shape[1]):
                alms[i, j, :, c] = hp.almxfl(alms[i, j, :, c], pixwin_in[0]) if j==0 else hp.almxfl(alms[i, j, :, c], pixwin_in[1])
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
    filtered_alms = np.array([hp.almxfl(alms[:,c], b_ell) for c in range(alms.shape[-1])]).T
    if alms.shape[0] == hp.Alm.getsize(lmax_out):
        return filtered_alms
    else:
        lmax_j = np.min([hp.Alm.getlmax(alms.shape[0]), lmax_out])
        alms_j = np.zeros((hp.Alm.getsize(lmax_out), alms.shape[-1]), dtype=complex)
        idx_lmax_in = np.array([hp.Alm.getidx(hp.Alm.getlmax(filtered_alms.shape[0]), ell, m) for ell in range(0, lmax_j+1) for m in range(ell + 1)])
        idx_lmax_out = np.array([hp.Alm.getidx(lmax_out, ell, m) for ell in range(0, lmax_j+1) for m in range(ell + 1)])
        alms_j[idx_lmax_out, :] = filtered_alms[idx_lmax_in, :]
        return alms_j

def _get_local_cov(input_maps, lmax, ilc_bias, b_ell=None, mask=None, reduce_bias=False):
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
            if mask is None:
                cov[i,k]=_get_local_cov_(input_maps[i], input_maps[k], pps, int(np.min([hp.npix2nside(input_maps.shape[1]),128])), reduce_bias=reduce_bias)
            else:
                cov[i,k]=_get_local_cov_(input_maps[i], input_maps[k], pps, int(hp.npix2nside(input_maps.shape[1])), reduce_bias=reduce_bias)
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

    stat_out = hp.alm2map(alm_s, nside_covar, lmax=lmax_stat, verbose=False)

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

    stat_out = hp.alm2map(alm_s, nside_covar, lmax=lmax_stat, verbose=False)

    return stat_out    

def _save_compsep_products(config: Configs, output_maps, compsep_run, nsim=None):
#    if config_dir_outputs[-1] != "/":
#        config_dir_outputs += "/"
    complete_path = os.path.join(f'{compsep_run["method"]}_{compsep_run["domain"]}_bias{compsep_run["ilc_bias"]}')
    if config.leakage_correction is not None:
        leak_def = (config.leakage_correction).split("_")[0] + (config.leakage_correction).split("_")[1] 
        if "_recycling" in config.leakage_correction:
            if "_iterations" in config.leakage_correction:
                iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
                leak_def += f'_iters{iterations}'
        complete_path += f"_{leak_def}"

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
        complete_path = os.path.join(complete_path, text_)
    path_out = os.path.join(config.path_outputs, complete_path)

    for c in range(output_maps.shape[-1]):
        path_c = os.path.join(path_out, f"{config.labels_outputs[c]}")
        os.makedirs(path_c, exist_ok=True)
        if nsim is None:
            hp.write_map(f"{path_c}/{config.field_out}_{config.labels_outputs[c]}_{config.fwhm_out}acm_ns{config.nside}.fits", output_maps[..., c], overwrite=True)
        else:
            hp.write_map(f"{path_c}/{config.field_out}_{config.labels_outputs[c]}_{config.fwhm_out}acm_ns{config.nside}_{str(nsim).zfill(5)}.fits", output_maps[..., c], overwrite=True)

    
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
          
        

