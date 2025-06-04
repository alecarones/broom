import healpy as hp
import numpy as np
import os 
import pymaster as nmt
import fnmatch

REMOTE = 'https://irsa.ipac.caltech.edu/data/Planck/release_2/'
import os.path as op
from astropy.utils.data import download_file
import astropy
from ._compsep import _load_outputs, _preprocess_mask
from .routines import _slice_data, obj_out_to_array, _slice_outputs
from .configurations import Configs
from types import SimpleNamespace

def _compute_spectra(config: Configs):

    """
    Compute the spectra from outputs of component separation.
    
    Parameters
    ----------
    config : Configs
        Configuration object containing parameters for spectra computation.
    
    Returns (if return_spectra is True)
    -------
    cls_ : SimpleNamespace
        Object containing computed spectra with attributes for each component.
        Each attribute is a numpy array with dimensions (nsim, ncases, nfields, nbins), 
        where ncases is the number of different component separation outputs provided in compute_spectra dictionary.
    """
    
    if not isinstance(config, Configs):
        raise TypeError("config must be an instance of Configs")

    if config.return_spectra:
        cls_ = SimpleNamespace()

    for nsim in range(config.nsim_start, config.nsim_start + config.nsims):
        if config.return_spectra:
            cls_sim = _compute_spectra_(config, nsim=nsim)
            for attr in vars(cls_sim):
                if not hasattr(cls_, attr):
                    setattr(cls_, attr, [])
                getattr(cls_, attr).append(getattr(cls_sim, attr))
        else:
            _compute_spectra_(config, nsim=nsim)

    if config.return_spectra:
        for attr in vars(cls_).keys():
            setattr(cls_, attr, np.array(getattr(cls_, attr)))
        return cls_


def _compute_spectra_(config: Configs, nsim=None):
    """
    Compute the spectra from outputs of component separation for simulation nsim.

    Parameters
    ----------
    config : Configs
        Configuration object containing parameters for spectra computation.

    nsim : int or str, optional
        Simulation number to compute spectra for. If an integer, it will be zero-padded to 5 digits.
        If a string, it should already be formatted as such. 
        Default is None, which means it will look for outputs with no label regarding simulation number.

    Returns (if return_spectra is True)
    -------
    cls_out_ : SimpleNamespace
        Object containing computed spectra with attributes for each component.
        Each attribute is a numpy array with dimensions (ncases, nfields, nbins), 
        where ncases is the number of different component separation outputs provided in compute_spectra dictionary.
    """

    if nsim is not None:
        if not isinstance(nsim, (int, str)):
            raise ValueError("Invalid value for nsim. It must be an integer or a string.")
        if isinstance(nsim, int):
            nsim = str(nsim).zfill(5)

    if config.spectra_comp not in ['anafast','namaster']:
        raise ValueError('spectra_comp must be either "anafast" or "namaster"')

    if config.return_spectra:
        cls_out_ = SimpleNamespace()

    for compute_cls in config.compute_spectra:
        compute_cls = _standardize_compute_cls(config, compute_cls)
        if config.return_spectra:
            cls_out = _cls_from_config(config, compute_cls, nsim=nsim)
            for attr in vars(cls_out):
                if not hasattr(cls_out_, attr):
                    setattr(cls_out_, attr, [])
                getattr(cls_out_, attr).append(getattr(cls_out, attr))
        else:
            _cls_from_config(config, compute_cls, nsim=nsim)

    if config.return_spectra:
        for attr in vars(cls_out_).keys():
            setattr(cls_out_, attr, np.array(getattr(cls_out_, attr)))
        return cls_out_

def _standardize_compute_cls(config: Configs, compute_cls):
    """
    Standardize the compute_cls dictionary to ensure it contains all necessary fields and has the correct structure.
    """

    if 'path_method' not in compute_cls:
        raise ValueError('compute_cls must contain a "path_method" field')
    if 'field_out' not in compute_cls:
        compute_cls['field_out'] = config.field_out
    if "components_for_cls" not in compute_cls:
        raise ValueError('compute_cls must contain a "components_for_cls" field')
    if "mask_type" not in compute_cls:
        compute_cls["mask_type"] = None
    if "apodize_mask" not in compute_cls:
        compute_cls["apodize_mask"] = None
    if compute_cls["apodize_mask"] is not None:
        if "smooth_mask" not in compute_cls:
            compute_cls["smooth_mask"] = 5.
    if compute_cls["mask_type"] is not None:
        if "fgres" in compute_cls["mask_type"]:
            if not "smooth_tracer" in compute_cls:
                compute_cls["smooth_tracer"] = 3.
            if "fsky" not in compute_cls:
                compute_cls["fsky"] = 1.
                print("fsky not defined in compute_spectra dictionary, setting it to 1.0. This will lead to no thresholding of the fgds residuals.")
    if config.spectra_comp == 'namaster':
        if "nmt_purify_B" not in compute_cls:
            compute_cls["nmt_purify_B"] = True
        if "nmt_purify_E" not in compute_cls:
            compute_cls["nmt_purify_E"] = False
    return compute_cls
    
def _cls_from_config(config: Configs, compute_cls, nsim=None):
    """
    Compute the spectra from outputs of component separation based on the configuration and compute_cls dictionary.
    
    Parameters
    ----------
    config : Configs
        Configuration object containing parameters for spectra computation.
    
    compute_cls : dict
        Dictionary containing parameters for spectra computation, including path_method, field_out, components_for_cls, mask_type, apodize_mask, smooth_mask, nmt_purify_B, nmt_purify_E.

    nsim : int or str, optional
        Simulation number to compute spectra for. If an integer, it will be zero-padded to 5 digits.
        If a string, it should already be formatted as such.
        Default is None, which means it will look for outputs with no label regarding simulation number.
    
    Returns
    -------
    cls_out : SimpleNamespace
        Object containing computed spectra with attributes for each component.
        Each attribute is a numpy array with dimensions (ncases, nfields, nbins), 
        where ncases is the number of different component separation outputs provided in compute_cls dictionary.
    """


    compute_cls["outputs"] = SimpleNamespace()

    compute_cls["path"] = os.path.join(config.path_outputs, compute_cls["path_method"])

    _check_fields_for_cls(compute_cls["field_out"],config.field_cls_out)
    compute_cls["field_cls_in"] = _get_fields_in_for_cls(compute_cls["field_out"],config.field_cls_out)
    
    for component in compute_cls["components_for_cls"]:
        if '/' in component:
            filename = os.path.join(compute_cls["path"], f"{component}/{compute_cls['field_out']}_{component.split('/')[0]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}")
            setattr(compute_cls["outputs"], component.split('/')[0], _load_outputs(filename,compute_cls["field_out"],nsim=nsim)) 
        else:
            filename = os.path.join(compute_cls["path"], f"{component}/{compute_cls['field_out']}_{component}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}")
            setattr(compute_cls["outputs"], component, _load_outputs(filename,compute_cls["field_out"],nsim=nsim)) 

    #if (config.field_out in ["TQU", "TEB"]) and (config.field_cls_out not in ["TE", "TB", "TEB"]):
    if len(compute_cls['field_out']) > 1:
        compute_cls["outputs"] = _slice_outputs(compute_cls["outputs"],compute_cls["field_out"],compute_cls["field_cls_in"])

    cls_out = _cls_from_maps(config, compute_cls, nsim=nsim)
    if config.save_spectra:
        save_spectra(config, cls_out, compute_cls, nsim=nsim)

    del compute_cls["outputs"]
    del compute_cls["path"]
    if "mask" in compute_cls:
        del compute_cls["mask"]

    if config.return_spectra:
        return cls_out

def save_spectra(config: Configs, cls_out, compute_cls, nsim=None):
    """
    Save the computed spectra to a file.
    """
    path_spectra = os.path.join(compute_cls["path"], 'spectra')
    mask_patterns = ['GAL*+fgres', 'GAL*0', 'GAL97', 'GAL99', 'fgres', 'config+fgres', 'config']
    if compute_cls["mask_type"] is None and config.mask_path is None:
        mask_name = 'fullsky'
    elif compute_cls["mask_type"] is None and config.mask_path is not None:
        mask_name = "fullpatch"
    elif any(fnmatch.fnmatch(compute_cls["mask_type"], pattern) for pattern in mask_patterns):
        if 'fgres' in compute_cls["mask_type"]:
            mask_name = compute_cls["mask_type"] + f"_fsky{compute_cls['fsky']}"
            if "smooth_tracer" in compute_cls:
                mask_name += f"_{compute_cls['smooth_tracer']}deg"
        elif compute_cls["mask_type"] == "config":
            mask_name = "fullpatch"
        else:
            mask_name = compute_cls["mask_type"]
    elif compute_cls["mask_type"] == "from_fits":
        if "mask_definition" not in compute_cls:
            compute_cls["mask_definition"] = "masks_from_fits"
        mask_name = compute_cls["mask_definition"]

    if compute_cls["apodize_mask"] is not None:
        mask_name += f"_apo{compute_cls['apodize_mask']}_{compute_cls['smooth_mask']}deg"

    path_spectra = os.path.join(path_spectra, mask_name)
    
    post_filename = f"_{nsim}" if nsim is not None else ""
    for component in compute_cls["components_for_cls"]:
        os.makedirs(os.path.join(path_spectra, component), exist_ok=True)
        if '/' in component:
            filename = os.path.join(path_spectra, f"{component}/{config.field_cls_out}_{component.split('/')[0]}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}{post_filename}.fits")
            hp.write_cl(filename, getattr(cls_out, component.split('/')[0]), overwrite=True) 
        else:
            filename = os.path.join(path_spectra, f"{component}/{config.field_cls_out}_{component}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}{post_filename}.fits")
            hp.write_cl(filename, getattr(cls_out, component), overwrite=True) 
        
def _cls_from_maps(config: Configs, compute_cls, nsim=None):
    compute_cls["mask"] = _get_mask(config, compute_cls, nsim=nsim)

    if compute_cls["apodize_mask"] is not None:
        compute_cls["mask"] = _smooth_masks(compute_cls["mask"], compute_cls["apodize_mask"], compute_cls["smooth_mask"])

    cls_out = _get_cls(config, compute_cls, nsim=nsim)

    return cls_out

def _get_cls(config: Configs, compute_cls, nsim=None):
    b_bin = nmt.NmtBin.from_lmax_linear(config.lmax, nlb=config.delta_ell,is_Dell=config.return_Dell)

    bls_beam = get_bls(config.nside, config.fwhm_out, config.lmax, config.field_cls_out, pixel_window_out=config.pixel_window_out)
    
    cls_out = SimpleNamespace()
    for attr in vars(compute_cls["outputs"]):
        setattr(cls_out, attr, [])

    if obj_out_to_array(compute_cls["outputs"]).ndim == 2:
        for idx, attr in enumerate(vars(compute_cls["outputs"])):
            if config.spectra_comp == 'anafast':
                getattr(cls_out, attr).append(b_bin.bin_cell(hp.anafast(getattr(compute_cls["outputs"], attr)*(compute_cls["mask"]),lmax=config.lmax,pol=False)/(np.mean((compute_cls["mask"])**2))/((bls_beam[0])**2)))
            elif config.spectra_comp == 'namaster':
                f = nmt.NmtField(compute_cls["mask"], [getattr(compute_cls["outputs"], attr)],beam=bls_beam[0],lmax=config.lmax,lmax_mask=config.lmax)
                if idx==0:
                    w00 = nmt.NmtWorkspace.from_fields(f, f, b_bin)
                getattr(cls_out, attr).append((nmt.compute_full_master(f, f, b_bin, workspace=w00))[0])
    else:
        if 'QU' not in compute_cls["field_cls_in"]:
            for field in range(obj_out_to_array(compute_cls["outputs"]).shape[1]):
                for idx, attr in enumerate(vars(compute_cls["outputs"])):
                    if config.spectra_comp == 'anafast':
                        getattr(cls_out, attr).append(b_bin.bin_cell(hp.anafast((getattr(compute_cls["outputs"], attr)[field])*(compute_cls["mask"][field]),lmax=config.lmax,pol=False)/(np.mean((compute_cls["mask"][field])**2))/((bls_beam[field])**2)))
                    elif config.spectra_comp == 'namaster':
                        f = nmt.NmtField(compute_cls["mask"][field], [getattr(compute_cls["outputs"], attr)[field]],beam=bls_beam[field],lmax=config.lmax,lmax_mask=config.lmax)
                        if idx==0:
                            w00 = nmt.NmtWorkspace.from_fields(f, f, b_bin)
                        getattr(cls_out, attr).append((nmt.compute_full_master(f, f, b_bin, workspace=w00))[0])
            if "EETE" in config.field_cls_out or "BBTE" in config.field_cls_out:
                for idx, attr in enumerate(vars(compute_cls["outputs"])):
                    if config.spectra_comp == 'anafast':
                        mask_ =  compute_cls["mask"][1] if np.mean(np.ceil(compute_cls["mask"][0])) > np.mean(np.ceil(compute_cls["mask"][1])) else compute_cls["mask"][0]
                        getattr(cls_out, attr).append(b_bin.bin_cell(hp.anafast((getattr(compute_cls["outputs"], attr)[0])*mask_,map2=(getattr(compute_cls["outputs"], attr)[1])*mask_,lmax=config.lmax,pol=False)/(np.mean(mask_**2))/(bls_beam[0]*bls_beam[1])))
                    elif config.spectra_comp == 'namaster':
                        f1 = nmt.NmtField(compute_cls["mask"][0], [getattr(compute_cls["outputs"], attr)[0]],beam=bls_beam[0],lmax=config.lmax,lmax_mask=config.lmax)
                        f2 = nmt.NmtField(compute_cls["mask"][1], [getattr(compute_cls["outputs"], attr)[1]],beam=bls_beam[1],lmax=config.lmax,lmax_mask=config.lmax)
                        if idx==0:
                            w00 = nmt.NmtWorkspace.from_fields(f1, f2, b_bin)
                        getattr(cls_out, attr).append((nmt.compute_full_master(f1, f2, b_bin, workspace=w00))[0])
            if "BBEB" in config.field_cls_out or "BTEEB" in config.field_cls_out:
                if "T" in compute_cls["field_cls_in"]:
                    field_E = 1
                    field_B = 2
                else:
                    field_E = 0
                    field_B = 1
                for idx, attr in enumerate(vars(compute_cls["outputs"])):
                    if config.spectra_comp == 'anafast':
                        mask_ =  compute_cls["mask"][field_B] if np.mean(np.ceil(compute_cls["mask"][field_E])) > np.mean(np.ceil(compute_cls["mask"][field_B])) else compute_cls["mask"][field_E]
                        getattr(cls_out, attr).append(b_bin.bin_cell(hp.anafast((getattr(compute_cls["outputs"], attr)[field_E])*mask_,map2=(getattr(compute_cls["outputs"], attr)[field_B])*mask_,lmax=config.lmax,pol=False)/(np.mean(mask_**2))/(bls_beam[field_E]*bls_beam[field_B])))
                    elif config.spectra_comp == 'namaster':
                        f1 = nmt.NmtField(compute_cls["mask"][field_E], [getattr(compute_cls["outputs"], attr)[field_E]],beam=bls_beam[field_E],lmax=config.lmax,lmax_mask=config.lmax)
                        f2 = nmt.NmtField(compute_cls["mask"][field_B], [getattr(compute_cls["outputs"], attr)[field_B]],beam=bls_beam[field_B],lmax=config.lmax,lmax_mask=config.lmax)
                        if idx==0:
                             w00 = nmt.NmtWorkspace.from_fields(f1, f2, b_bin)
                        getattr(cls_out, attr).append((nmt.compute_full_master(f1, f2, b_bin, workspace=w00))[0])
            if "BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out:
                if "E" in compute_cls["field_cls_in"]:
                    field_B = 2
                else:
                    field_B = 1
                for idx, attr in enumerate(vars(compute_cls["outputs"])):
                    if config.spectra_comp == 'anafast':
                        mask_ =  compute_cls["mask"][field_B] if np.mean(np.ceil(compute_cls["mask"][0])) > np.mean(np.ceil(compute_cls["mask"][field_B])) else compute_cls["mask"][0]
                        getattr(cls_out, attr).append(b_bin.bin_cell(hp.anafast((getattr(compute_cls["outputs"], attr)[0])*mask_,map2=(getattr(compute_cls["outputs"], attr)[field_B])*mask_,lmax=config.lmax,pol=False)/(np.mean(mask_**2))/(bls_beam[0]*bls_beam[field_B])))
                    elif config.spectra_comp == 'namaster':
                        f1 = nmt.NmtField(compute_cls["mask"][0], [getattr(compute_cls["outputs"], attr)[0]],beam=bls_beam[0],lmax=config.lmax,lmax_mask=config.lmax)
                        f2 = nmt.NmtField(compute_cls["mask"][field_B], [getattr(compute_cls["outputs"], attr)[field_B]],beam=bls_beam[field_B],lmax=config.lmax,lmax_mask=config.lmax)
                        if idx==0:
                            w00 = nmt.NmtWorkspace.from_fields(f1, f2, b_bin)
                        getattr(cls_out, attr).append((nmt.compute_full_master(f1, f2, b_bin, workspace=w00))[0])
        else:
            if "T" in compute_cls["field_cls_in"]:
                mask_ =  compute_cls["mask"][1] if np.mean(np.ceil(compute_cls["mask"][0])) > np.mean(np.ceil(compute_cls["mask"][1])) else compute_cls["mask"][0]
                field_Q = 1
                if "EE" in config.field_cls_out:
                    field_E = 1
                    if "BB" in config.field_cls_out:
                        field_B = 2
                else:
                    field_B = 1
                beam_nmt = bls_beam[0]
            else:
                mask_ =  compute_cls["mask"][0]
                field_Q = 0
                if "EE" in config.field_cls_out:
                    field_E = 0
                    if "BB" in config.field_cls_out:
                        field_B = 1
                else:
                    field_B = 0
                beam_nmt = hp.gauss_beam(np.radians(config.fwhm_out/60.), lmax=config.lmax, pol=False)
                if config.pixel_window_out:
                    beam_nmt *= hp.pixwin(config.nside, lmax=config.lmax, pol=False)

            for idx, attr in enumerate(vars(compute_cls["outputs"])):
                if "T" in compute_cls["field_cls_in"]:
#                    T_map, Q_map, U_map = getattr(compute_cls["outputs"], attr)[0] * np.ceil(compute_cls["mask"][0]), getattr(compute_cls["outputs"], attr)[1] * np.ceil(compute_cls["mask"][1]), getattr(compute_cls["outputs"], attr)[2] * np.ceil(compute_cls["mask"][2])
                    T_map, Q_map, U_map = getattr(compute_cls["outputs"], attr)[0], getattr(compute_cls["outputs"], attr)[1], getattr(compute_cls["outputs"], attr)[2]
                else:
#                    T_map, Q_map, U_map = np.zeros_like(getattr(compute_cls["outputs"], attr)[0]), getattr(compute_cls["outputs"], attr)[0] * np.ceil(compute_cls["mask"][0]), getattr(compute_cls["outputs"], attr)[1] * np.ceil(compute_cls["mask"][1])
                    T_map, Q_map, U_map = np.zeros_like(getattr(compute_cls["outputs"], attr)[0]), getattr(compute_cls["outputs"], attr)[0], getattr(compute_cls["outputs"], attr)[1]

                if config.spectra_comp == 'anafast':
                    cls_s2 = hp.anafast([T_map * compute_cls["mask"][0], Q_map * compute_cls["mask"][field_Q], U_map * compute_cls["mask"][field_Q+1]], lmax=config.lmax, pol=True)
                    if "TT" in config.field_cls_out:
                        getattr(cls_out, attr).append(b_bin.bin_cell(cls_s2[0] / np.mean((compute_cls["mask"][0])**2) / (bls_beam[0]**2)))
                    if "EE" in config.field_cls_out:
                        getattr(cls_out, attr).append(b_bin.bin_cell(cls_s2[1] / np.mean((compute_cls["mask"][field_Q])**2) / (bls_beam[field_E]**2)))
                    if "BB" in config.field_cls_out:
                        getattr(cls_out, attr).append(b_bin.bin_cell(cls_s2[2] / np.mean((compute_cls["mask"][field_Q])**2) / (bls_beam[field_B]**2)))
                    if any(x in config.field_cls_out for x in ["EETE", "BBTE", "BBEB", "BTEEB", "BBTB", "EBTB"]):
                        cls_s2_cross = hp.anafast([T_map * mask_, Q_map * mask_, U_map * mask_], lmax=config.lmax, pol=True) / np.mean(mask_**2)
                        if "EETE" in config.field_cls_out or "BBTE" in config.field_cls_out:
                            getattr(cls_out, attr).append(b_bin.bin_cell(cls_s2_cross[3]/(bls_beam[0]*bls_beam[1])))
                        if "BBEB" in config.field_cls_out or "BTEEB" in config.field_cls_out:
                            getattr(cls_out, attr).append(b_bin.bin_cell(cls_s2_cross[4]/(bls_beam[field_E]*bls_beam[field_B])))
                        if "BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out:
                            getattr(cls_out, attr).append(b_bin.bin_cell(cls_s2_cross[5]/(bls_beam[0]*bls_beam[field_B])))
                elif config.spectra_comp == 'namaster':
                    if "TT" in config.field_cls_out:
                        f_0 = nmt.NmtField(compute_cls["mask"][0], [T_map], beam=bls_beam[0],lmax=config.lmax,lmax_mask=config.lmax)
                        if idx==0:
                            w00 = nmt.NmtWorkspace.from_fields(f_0, f_0, b_bin)
                        getattr(cls_out, attr).append((nmt.compute_full_master(f_0, f_0, b_bin, workspace=w00))[0])
                    f2 = nmt.NmtField(compute_cls["mask"][field_Q], [Q_map, U_map], purify_b=compute_cls["nmt_purify_B"], purify_e=compute_cls["nmt_purify_E"], beam=beam_nmt, lmax=config.lmax, lmax_mask=config.lmax)
                    if idx==0:
                        w22 = nmt.NmtWorkspace.from_fields(f2, f2, b_bin)
                    if "EE" in config.field_cls_out:
                        getattr(cls_out, attr).append(w22.decouple_cell(nmt.compute_coupled_cell(f2, f2))[0])
                    if "BB" in config.field_cls_out:
                        getattr(cls_out, attr).append(w22.decouple_cell(nmt.compute_coupled_cell(f2, f2))[3])
                    if ("EETE" in config.field_cls_out or "BBTE" in config.field_cls_out) or ("BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out):
                        if idx==0:
                            w02 = nmt.NmtWorkspace.from_fields(f_0, f2, b_bin)
                    if "EETE" in config.field_cls_out or "BBTE" in config.field_cls_out:
                        getattr(cls_out, attr).append(w02.decouple_cell(nmt.compute_coupled_cell(f_0, f2))[0])
                    if "BBEB" in config.field_cls_out or "BTEEB" in config.field_cls_out:
                        getattr(cls_out, attr).append(w22.decouple_cell(nmt.compute_coupled_cell(f2, f2))[1])
                    if "BBTB" in config.field_cls_out or "EBTB" in config.field_cls_out:
                        getattr(cls_out, attr).append(w02.decouple_cell(nmt.compute_coupled_cell(f_0, f2))[1])

    for attr in vars(cls_out).keys():
        setattr(cls_out, attr, np.array(getattr(cls_out, attr)))
    return cls_out
            
def get_bls(nside, fwhm, lmax, field_cls_out, pixel_window_out=True):
    bls_beam = []

    bl_ = hp.gauss_beam(np.radians(fwhm/60.), lmax=lmax, pol=True).T
    if pixel_window_out:
        pw_ = hp.pixwin(nside, lmax=lmax, pol=True)
        bl_[0] *= pw_[0]
        bl_[1] *= pw_[1]
        bl_[2] *= pw_[1]

    if 'TT' in field_cls_out:
        bls_beam.append(bl_[0])
    if 'EE' in field_cls_out:
        bls_beam.append(bl_[1])
    if 'BB' in field_cls_out:
        bls_beam.append(bl_[2])

    bls_beam = np.array(bls_beam)
    if bls_beam.ndim == 1:
        bls_beam = bls_beam[np.newaxis, :]

    return bls_beam

def _get_mask(config: Configs, compute_cls, nsim=None):
    n_fields_in = obj_out_to_array(compute_cls["outputs"]).shape[-2] if obj_out_to_array(compute_cls["outputs"]).ndim == 3 else 1
    npix = obj_out_to_array(compute_cls["outputs"]).shape[-1]
    
    mask_patterns = ['GAL*+fgres', 'GAL*0', 'GAL97', 'GAL99', 'fgres', 'config+fgres', 'config']

    if compute_cls["mask_type"] is None and config.mask_path is None:
        mask_spectra = np.ones(npix) if n_fields_in == 1 else np.ones((n_fields_in, npix))
    elif compute_cls["mask_type"] is None and config.mask_path is not None:
        mask_spectra = _preprocess_mask(hp.read_map(config.mask_path, field=0), config.nside) if n_fields_in == 1 else np.repeat(_preprocess_mask(hp.read_map(config.mask_path, field=0), config.nside)[np.newaxis, :], n_fields_in, axis=0)
    elif compute_cls["mask_type"] == "from_fits":
        if "mask_path" not in compute_cls:
            raise ValueError('mask_path must be defined in the compute_cls dictionary if mask_type is "from_fits"')
        if not isinstance(compute_cls["mask_path"], str):
            raise ValueError('mask_path must be a string.')
        if n_fields_in == 1:
            mask_spectra = hp.read_map(compute_cls["mask_path"], field=0)
        else:
            mask_spectra = np.zeros((n_fields_in, npix))
            mask_spectra[0] = hp.read_map(compute_cls["mask_path"], field=0)
            if compute_cls["field_cls_in"] == "TQU":
                try:
                    mask_spectra[1] = hp.read_map(compute_cls["mask_path"], field=1)
                    mask_spectra[2] = hp.read_map(compute_cls["mask_path"], field=1)
                except IndexError:
                    mask_spectra[1] = hp.read_map(compute_cls["mask_path"], field=0)
                    mask_spectra[2] = hp.read_map(compute_cls["mask_path"], field=0)
            elif compute_cls["field_cls_in"] in ["QU","QU_E","QU_B"]:
                mask_spectra[1] = hp.read_map(compute_cls["mask_path"], field=0)
            else:
                for i in range(1,n_fields_in):
                    try:
                        mask_spectra[i] = hp.read_map(compute_cls["mask_path"], field=i)
                    except IndexError:
                        mask_spectra[i] = hp.read_map(compute_cls["mask_path"], field=0)
    elif any(fnmatch.fnmatch(compute_cls["mask_type"], pattern) for pattern in mask_patterns):
        if 'GAL' in compute_cls["mask_type"]:
            gal_masks_list = ['GAL20','GAL40','GAL60','GAL70','GAL80','GAL90','GAL97','GAL99']
            if compute_cls["mask_type"][:5] not in gal_masks_list:
                raise ValueError('GAL mask must be one of GAL20, GAL40, GAL60, GAL70, GAL80, GAL90, GAL97, GAL99')
            mask_init = hp.ud_grade(get_planck_mask(0, field=np.argwhere(np.array(gal_masks_list) == compute_cls["mask_type"][:5])[0][0], nside=512),hp.npix2nside(npix)) == 1.
        elif 'config' in compute_cls["mask_type"]:
            if config.mask_path is None:
                mask_init = np.ones(npix)
            else:
                mask_init = hp.read_map(config.mask_path, field=0) == 1.
        else:
            mask_init = np.ones(npix)

        if 'fgres' in compute_cls["mask_type"]:
            if not "fsky" in compute_cls:
                raise ValueError('fsky must be defined in the config if fgres is used in mask_type')
            if not hasattr(compute_cls["outputs"], 'fgds_residuals'):
                fgres = SimpleNamespace()
                filename = os.path.join(compute_cls["path"], f"fgds_residuals/{compute_cls['field_out']}_fgds_residuals_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}")
                setattr(fgres, 'total', _load_outputs(filename,compute_cls["field_out"],nsim=nsim)) 
                fgres = _slice_data(fgres,compute_cls["field_out"],compute_cls["field_cls_in"])
                mask_spectra = get_threshold_mask(fgres.total,mask_init,compute_cls["field_cls_in"],compute_cls["fsky"],config.lmax,smooth_tracer=compute_cls["smooth_tracer"])
            else:
                mask_spectra = get_threshold_mask(compute_cls["outputs"].fgds_residuals,mask_init,compute_cls["field_cls_in"],compute_cls["fsky"],config.lmax,smooth_tracer=compute_cls["smooth_tracer"])
        else:
            mask_spectra = mask_init if n_fields_in == 1 else np.repeat(mask_init[np.newaxis, :], n_fields_in, axis=0)

    if 'mask_spectra' not in locals():
        raise ValueError('Mask not defined. Please check the mask_type and mask_path in the compute_cls dictionary.')

    return mask_spectra

def get_threshold_mask(map_,mask_init,field_cls_in,fsky,lmax,smooth_tracer=3.):
    fsky_in = np.mean(mask_init**2)
    if fsky > fsky_in:
        if map_.ndim ==1:
            return mask_init
        else:
            return np.repeat(mask_init[np.newaxis, :], map_.shape[0], axis=0)
    else:
        npix = map_.shape[-1]
        npix_mask = int((fsky_in - fsky) * npix)
        if map_.ndim == 1:
            mask_spectra = threshold_scalar_tracer(map_,mask_init,npix_mask,lmax,smooth_tracer=smooth_tracer)
        else:
            mask_spectra = np.ones((map_.shape[0], npix))
            if field_cls_in == "TQU":
                mask_spectra[0] = threshold_scalar_tracer(map_[0],mask_init,npix_mask,lmax,smooth_tracer=smooth_tracer)
                mask_spectra_P = threshold_P_tracer(map_[1:],mask_init,npix_mask,lmax,smooth_tracer=smooth_tracer)
                mask_spectra[1] = mask_spectra_P
                mask_spectra[2] = mask_spectra_P
            elif field_cls_in == "QU":
                mask_spectra_P = threshold_P_tracer(map_,mask_init,npix_mask,lmax,smooth_tracer=smooth_tracer)
                mask_spectra[0] = mask_spectra_P
                mask_spectra[1] = mask_spectra_P
            else:
                for i in range(map_.shape[0]):
                    mask_spectra[i] = threshold_scalar_tracer(map_[i],mask_init,npix_mask,lmax,smooth_tracer=smooth_tracer)
        return mask_spectra

def _smooth_masks(mask, apodization, smooth_mask):
    if apodization not in ["gaussian", "gaussian_nmt", "C1", "C2"]:
        raise ValueError('apodization must be either "gaussian", "gaussian_nmt", "C1" or "C2"')
    if mask.ndim == 1:
        mask_out = _smooth_mask(mask, apodization, smooth_mask)
    else:
        mask_out = []
        for i in range(mask.shape[0]):
            mask_out.append(_smooth_mask(mask[i], apodization, smooth_mask))
        mask_out = np.array(mask_out)
    return mask_out

def _smooth_mask(mask_in, apodization, smooth_mask):
    if apodization == "gaussian":
        mask_out = hp.smoothing(mask_in,fwhm=np.radians(smooth_mask))
    elif apodization == "gaussian_nmt":
        mask_out = nmt.mask_apodization(mask_in, smooth_mask, apotype="Smooth")
    elif apodization in ["C1", "C2"]:
        mask_out = nmt.mask_apodization(mask_in, smooth_mask, apotype=apodization)
    return np.array(mask_out)

def threshold_scalar_tracer(map_,mask_in,npix_mask,lmax,smooth_tracer=3.):
    mask_spectra = np.ones_like(map_)
#    idx_mask = np.argsort(np.absolute(hp.smoothing(map_,fwhm=np.radians(smooth_tracer),lmax=lmax,pol=False))*mask_in)[-npix_mask:]  #
    idx_mask = np.argsort(hp.smoothing(np.absolute(map_),fwhm=np.radians(smooth_tracer),lmax=lmax,pol=False)*mask_in)[-npix_mask:]
    mask_spectra[idx_mask]=0.
    mask_spectra[mask_in==0.]=0.
    return mask_spectra

def threshold_P_tracer(map_,mask_in,npix_mask,lmax,smooth_tracer=3.):
    mask_spectra = np.ones_like(map_[0])
    map_ = hp.smoothing([0.*map_,map_[0],map_[1]],fwhm=np.radians(deg_smooth_tracer),lmax=lmax,pol=True)[1:]
    map_P = np.sqrt((map_[0])**2 + (map_[1])**2)
    idx_mask = np.argsort(map_P*mask_in)[-npix_mask:]  #
    mask_spectra[idx_mask]=0.
    mask_spectra[mask_in==0.]=0.
    return mask_spectra

def _get_processed_dir():
    processed_dir = op.join(astropy.config.get_cache_dir(),
                            'processed', 'planck')
    if not op.exists(processed_dir):
        os.makedirs(processed_dir)

    return processed_dir

def get_planck_mask(apo=5, nside=2048, field=3, info=False):
    remote_url = op.join(REMOTE, 'ancillary-data/masks',
                         f'HFI_Mask_GalPlane-apo{apo}_2048_R2.00.fits')
    remote_file = download_file(remote_url, cache=True)
    if info:
        return hp.read_map(remote_file, field=[], h=True)

    local_file = op.join(_get_processed_dir(),
                         f'HFI_Mask_GalPlane-apo{apo}_{nside}_R2.00.fits')
    try:
        output = hp.read_map(local_file, field=field)
    except IOError:
        output = hp.read_map(remote_file, field=None)
        output = hp.ud_grade(output, nside)
        hp.write_map(local_file, output)
        output = output[field]
    return output


def _get_fields_in_for_cls(field_out,field_cls_out):
    if field_out == "TQU":
        if field_cls_out in ["EE", "BB", "EEBB", "EEBBEB"]:
            field_cls_in = "QU"
        elif field_cls_out == "TT":
            field_cls_in = "T"
        else:
            field_cls_in = "TQU"
    elif field_out in ['T', 'E', 'B', 'QU', 'QU_E', 'QU_B']:
        field_cls_in = field_out
    elif field_out in ["TEB", "EB"]:
        field_cls_in = ""
        if ("TT" in field_cls_out) and (field_out == "TEB"):
            field_cls_in += "T"
        if "EE" in field_cls_out:
            field_cls_in += "E"
        if "BB" in field_cls_out:
            field_cls_in += "B"
    return field_cls_in

def _check_fields_for_cls(field_out,field_cls_out):
    if field_out not in ['T','E','B','QU','QU_E','QU_B','EB','TQU','TEB']:
        raise ValueError('Invalid field_out.')
    if field_cls_out not in ['TT','EE','BB','TTEE','TTEETE','TTBB','TTBBTB','EEBB','EEBBEB','TTEEBB','TTEEBBTEEBTB', ]:
        raise ValueError('Invalid field_cls_out.')
    if field_out in ['T','E','B']:
        if field_cls_out != f"{field_out}{field_out}":
            raise ValueError('Invalid field_cls_out. It must be the same as field_out for the given field_out.')
    if field_out == "QU" or field_out == "EB":
        if field_cls_out not in ["EE", "BB", "EEBB", "EEBBEB"]:
            raise ValueError('Invalid field_cls_out. It must be "EE", "BB", "EEBB" or "EEBBEB" if field_out is "QU".')
    if field_out == "QU_E":
        if field_cls_out != "EE":
            raise ValueError('Invalid field_cls_out. It must be "EE" if field_out is "QU_E".')
    if field_out == "QU_B":
        if field_cls_out != "BB":
            raise ValueError('Invalid field_cls_out. It must be "BB" if field_out is "QU_B".')
    if field_out == "TQU" or field_out == "TEB":
        if field_cls_out not in ["TT", "EE", "BB", "TTEE", "TTEETE", "TTBB", "TTBBTB", "EEBB", "EEBBEB", "TTEEBB", "TTEEBBTEEBTB"]:
            raise ValueError('Invalid field_cls_out. It must be "TT", "EE", "BB", "TTEE", "TTEETE", "TTBB", "TTBBTB", "EEBB", "EEBBEB", "TTEEBB" or "TTEEBBTEEBTB" if field_out is "TQU" or "TEB".')