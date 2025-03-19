import healpy as hp
import numpy as np
import os 
import pymaster as nmt

REMOTE = 'https://irsa.ipac.caltech.edu/data/Planck/release_2/'
import os.path as op
from astropy.utils.data import download_file
import astropy
from _compsep import _load_outputs

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

def _cls_from_files(config: Configs, path, mask, lmax, fwhm, bin_l=1, spectra_comp='anafast', return_Dell=False):

def compute_cls(maps,fsky,lmax,mask_type='PTEP',spectra_comp='anafast',bin_l=1,load_mask=False,save_mask=False,return_Dell=False,field_mp=2,deg_smooth_mask=0.,deg_smooth_tracer=3.):
    if spectra_comp not in ['anafast','namaster']:
        raise ValueError('spectra_comp must be either "anafast" or "namaster"')
    if mask_type not in ['PTEP','fully_ideal']:
        raise ValueError('mask_type must be either "PTEP" or "fully_ideal"')
    if mask_type=='PTEP' and (fsky >= 0.59):
        print('PTEP mask with fsky > 0.6 uses Planck Galactic mask with sky fraction equal to 60%.')

    outputs = load_outputs(dir_maps,nsim,fwhm,nside)

    if not load_mask:
        mask_spectra = _get_mask(outputs[1],fsky,lmax,mask_type,field_mp=field_mp,deg_smooth_tracer=deg_smooth_tracer)
        if save_mask:
            maskname = dir_maps+f'/mask_{mask_type}_ns{nside}_fsky{fsky}_{str(nsim).zfill(4)}.fits'
            hp.write_map(maskname, mask_spectra, overwrite=True)
    else:
        maskname = dir_maps+f'/mask_{mask_type}_ns{nside}_fsky{fsky}_{str(nsim).zfill(4)}.fits'
        mask_spectra = hp.read_map(maskname, verbose=False)
    
    if deg_smooth_mask > 0.:
        mask_spectra = hp.smoothing(mask_spectra,fwhm=np.radians(deg_smooth_mask),lmax=lmax,pol=False,verbose=False)

    cls = _get_cls(outputs,mask_spectra,lmax,fwhm,bin_l=bin_l,spectra_comp=spectra_comp,return_Dell=return_Dell)

    return cls

def load_outputs(dir_maps,nsim,fwhm,nside):
    outputs = np.load(dir_maps+f'/outputs_mcnilc_{fwhm}acm_ns{nside}_{str(nsim).zfill(4)}.npy')
    return outputs

def _get_cls(outputs,mask_spectra,lmax,fwhm,bin_l=1,spectra_comp='anafast',return_Dell=False):
    b_bin = nmt.NmtBin(nlb=bin_l,lmax=lmax,is_Dell=return_Dell)  
        
    nside = hp.get_nside(outputs[0])
    bl=hp.gauss_beam(np.radians(fwhm/60.), lmax=3*nside, pol=True)[:,2]
    
    if spectra_comp=='namaster':
        f_out=nmt.NmtField(mask_spectra, [outputs[0]], beam=bl)
        f_fres=nmt.NmtField(mask_spectra, [outputs[1]], beam=bl)
        f_nres=nmt.NmtField(mask_spectra, [outputs[2]], beam=bl)
        f_cmb=nmt.NmtField(mask_spectra, [outputs[0]-outputs[1]-outputs[2]], beam=bl)
    
        wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(f_out, f_out, b_bin)

        cls_out = (nmt.compute_full_master(f_out, f_out, b_bin, workspace=wsp))[0] 
        cls_fres = (nmt.compute_full_master(f_fres, f_fres, b_bin, workspace=wsp))[0] 
        cls_nres = (nmt.compute_full_master(f_nres, f_nres, b_bin, workspace=wsp))[0] 
        cls_cmb = (nmt.compute_full_master(f_cmb, f_cmb, b_bin, workspace=wsp))[0] 
    
    elif spectra_comp=='anafast':
        fsky = np.mean(mask_spectra)
        bl = np.copy(bl[:lmax+1])

        cls_out = hp.anafast((outputs[0])*mask_spectra,lmax=lmax,pol=False)/fsky/(bl**2)
        cls_fres = hp.anafast((outputs[1])*mask_spectra,lmax=lmax,pol=False)/fsky/(bl**2)
        cls_nres = hp.anafast((outputs[2])*mask_spectra,lmax=lmax,pol=False)/fsky/(bl**2)
        cls_cmb = hp.anafast((outputs[0]-outputs[1]-outputs[2])*mask_spectra,lmax=lmax,pol=False)/fsky/(bl**2)

        cls_out = b_bin.bin_cell(cls_out)
        cls_fres = b_bin.bin_cell(cls_fres)
        cls_nres = b_bin.bin_cell(cls_nres)
        cls_cmb = b_bin.bin_cell(cls_cmb)
        
    return np.array([cls_out,cls_fres,cls_nres,cls_cmb])

def _get_mask(map_,fsky,lmax,mask_type, field_mp=2, deg_smooth_tracer=3.):
    if mask_type == 'PTEP':
        mask_planck = get_planck_mask(0, field=field_mp, nside=hp.get_nside(map_)) == 1.
        mask_spectra = get_mask_PTEP(map_,mask_planck,fsky,lmax,deg_smooth_tracer=deg_smooth_tracer)
    elif mask_type == 'fully_ideal':
        mask_spectra = get_mask_PTEP(map_,np.ones_like(map_),fsky,lmax,deg_smooth_tracer=deg_smooth_tracer)
    return mask_spectra

def get_mask_PTEP(map_,mask_in,fsky,lmax, deg_smooth_tracer=3.):
    nside = hp.get_nside(map_)
    npix = hp.nside2npix(nside)
    fsky_in = np.mean(mask_in)
    npix_mask = int((fsky_in - fsky) * npix)

    mask_spectra=np.ones(npix)
    idx_mask = np.argsort(np.absolute(hp.smoothing(map_,fwhm=np.radians(deg_smooth_tracer),lmax=lmax,pol=False,verbose=False))*mask_in)[-npix_mask:]  #
    mask_spectra[idx_mask]=0.
    mask_spectra[mask_in==0.]=0.
    
    return mask_spectra



