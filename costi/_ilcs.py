import numpy as np
import healpy as hp
from .configurations import Configs
from .routines import _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering,\
                      _get_local_cov, _EB_to_QU, _E_to_QU, _B_to_QU, _save_compsep_products
from ._seds import _get_CMB_SED

def ilc(config: Configs, input_alms, compsep_run, nsim=None):
    output_maps = _ilc(config, input_alms, compsep_run)
    
    output_maps = _ilc_post_processing(config, output_maps, compsep_run)

    if config.save_compsep_products:
        _save_compsep_products(config, output_maps, compsep_run, nsim=nsim)
    if config.return_compsep_products:
        return output_maps

def _ilc_post_processing(config: Configs, output_maps, compsep_run):
    if (output_maps.ndim==3) and ((output_maps.shape[0]==2 and config.field_out == "QU") or (output_maps.shape[0]==3 and config.field_out == "TQU")):
        outputs = np.zeros_like(output_maps)
        for c in range(output_maps.shape[-1]):
            outputs[:,:,c] = _EB_to_QU(output_maps[:,:,c],config.lmax)
        if hasattr(compsep_run, "mask"):
            outputs[:,compsep_run["mask"] == 0.,:] = 0.
        return outputs
    elif (output_maps.ndim==2) and (config.field_out in ["QU_E", "QU_B"]):
        output = np.zeros((2, output_maps.shape[0], output_maps.shape[-1]))
        for c in range(output_maps.shape[-1]):
            if config.field_out == "QU_E":
                output[:,:,c] = _E_to_QU(output_maps[:,c],config.lmax)
            elif config.field_out=="QU_B":
                output[:,:,c] = _B_to_QU(output_maps[:,c],config.lmax)
        if hasattr(compsep_run, "mask"):
            output[:,compsep_run["mask"] == 0.,:] = 0.
        return output
    else:
        if hasattr(compsep_run, "mask"):
            output_maps[compsep_run["mask"] == 0.,:] = 0.
        return output_maps

def _ilc(config: Configs, input_alms, compsep_run):
    if input_alms.ndim == 4:
        output_maps = np.zeros((input_alms.shape[1], 12 * config.nside**2, input_alms.shape[3]))
        for i in range(input_alms.shape[1]):
            output_maps[i] = _ilc_scalar(config, input_alms[:, i, :, :], compsep_run)
    elif input_alms.ndim == 3:
        output_maps = _ilc_scalar(config, input_alms, compsep_run)
    return output_maps

def _ilc_scalar(config: Configs, input_alms, compsep_run):
    if compsep_run["domain"] == "pixel":
        output_maps = _ilc_pixel(config, input_alms, compsep_run)
    elif compsep_run["domain"] == "needlet":
        output_maps = _ilc_needlet(config, input_alms, compsep_run)
    elif compsep_run["domain"] == "harmonic":
        output_maps = _hilc(config, input_alms, compsep_run)
    return output_maps

def _ilc_needlet(config: Configs, input_alms, compsep_run):
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    if compsep_run["b_squared"]:
        b_ell = b_ell**2
    
    output_alms = np.zeros((input_alms.shape[1], input_alms.shape[-1]), dtype=complex)
    for j in range(b_ell.shape[0]):
        output_alms += _ilc_needlet_j(config, input_alms, compsep_run, b_ell[j])
    
    output_maps = np.array([hp.alm2map(np.ascontiguousarray(output_alms[:, c]), config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out) for c in range(input_alms.shape[-1])]).T
    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.,:] = 0.
    return output_maps
        
def _ilc_needlet_j(config: Configs, input_alms, compsep_run, b_ell):
    if "mask" in compsep_run:
        nside_, lmax_ = config.nside, config.lmax
    else:
        if compsep_run["adapt_nside"]:
            nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
        else:
            nside_, lmax_ = config.nside, config.lmax

    good_channels_nl = _get_good_channels_nl(config, b_ell)

    input_maps_nl = np.zeros((good_channels_nl.shape[0], 12 * nside_**2, input_alms.shape[-1]))
    for n, channel in enumerate(good_channels_nl):
        input_alms_j = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        if "mask" in compsep_run:
            input_maps_nl[n] = np.array([(hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False) * compsep_run["mask"]) for c in range(input_alms.shape[-1])]).T            
        else:
            input_maps_nl[n] = np.array([hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False) for c in range(input_alms.shape[-1])]).T
    output_maps_nl = _ilc_maps(config, input_maps_nl, compsep_run, b_ell)
    del input_maps_nl
    output_alms_nl = np.array([hp.map2alm(output_maps_nl[:,c], lmax=lmax_, pol=False) for c in range(output_maps_nl.shape[-1])]).T

    if compsep_run["b_squared"]:
        output_alms_j = _needlet_filtering(output_alms_nl, np.ones(lmax_+1), config.lmax)
    else:
        output_alms_j = _needlet_filtering(output_alms_nl, b_ell[:lmax_+1], config.lmax)
    return output_alms_j

def _get_good_channels_nl(config, b_ell, threshold=1e-5):
    freqs_nl = []
    for i in range(len(config.instrument.fwhm)):
        bl_in = hp.gauss_beam(np.radians(config.instrument.fwhm[i]/60.), lmax=config.lmax,pol=False)
        bl_out = hp.gauss_beam(np.radians(config.fwhm_out/60.), lmax=config.lmax,pol=False)
        if np.sum((bl_in/bl_out)[b_ell>1e-2] < threshold) == 0.:
            freqs_nl.append(i)
    return np.array(freqs_nl)

def _ilc_pixel(config: Configs, input_alms, compsep_run):
    input_maps = np.zeros((input_alms.shape[0], 12 * config.nside**2, input_alms.shape[-1]))
    for n in range(input_alms.shape[0]):
        if "mask" in compsep_run:
            input_maps[n] = np.array([(hp.alm2map(np.ascontiguousarray(input_alms[n, :, c]), config.nside, lmax=config.lmax, pol=False) * compsep_run["mask"]) for c in range(input_alms.shape[-1])]).T
        else:
            input_maps[n] = np.array([hp.alm2map(np.ascontiguousarray(input_alms[n, :, c]), config.nside, lmax=config.lmax, pol=False) for c in range(input_alms.shape[-1])]).T
    output_maps = _ilc_maps(config, input_maps, compsep_run, np.ones(config.lmax+1))
    if config.pixel_window_out:
        for c in range(output_maps.shape[1]):
            alm_out = hp.map2alm(output_maps[:,c],lmax=config.lmax, pol=False)
            output_maps[:,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=False, pixwin=True)
    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.,:] = 0.
    return output_maps

def _ilc_maps(config: Configs, input_maps, compsep_run, b_ell):
    A_cmb = _get_CMB_SED(config.instrument.frequency, units=config.units)

    cov = get_ilc_cov(input_maps[...,0], config.lmax, compsep_run, b_ell)
    inv_cov = get_inv_cov(cov)
    del cov
    w_ilc = get_ilc_weights(A_cmb, inv_cov, input_maps, compsep_run)
    del inv_cov

    if compsep_run["ilc_bias"] == 0.:
        output_maps = np.einsum('i,ijk->jk', w_ilc, input_maps)
    else:
        output_maps = np.einsum('ij,ijk->jk', w_ilc, input_maps)
    return output_maps

def get_ilc_weights(A_cmb, inv_cov, input_maps, compsep_run):
    if compsep_run["ilc_bias"] == 0.:
#        w_ilc = np.sum(inv_cov,axis=1)/np.sum(inv_cov)
        w_ilc = (A_cmb.T @ inv_cov) / (A_cmb.T @ inv_cov @ A_cmb) 
    else:
        w_ilc=np.zeros((input_maps.shape[0],input_maps.shape[1]))
        AT_invC = np.einsum('j,ijk->ik', A_cmb, inv_cov) # np.sum(inv_cov,axis=1)
        AT_invC_A = np.einsum('j,ijk, i->k', A_cmb, inv_cov, A_cmb) #np.sum(inv_cov,axis=(0,1))
        for i in range(input_maps.shape[0]):
            if "mask" in compsep_run:
                w_ilc[i, compsep_run["mask"] > 0.] = AT_invC[i]/AT_invC_A
            else:
                w_ilc[i]=hp.ud_grade(AT_invC[i]/AT_invC_A,hp.npix2nside(input_maps.shape[1]))
    return w_ilc

def get_ilc_cov(input_maps, lmax, compsep_run, b_ell):
    if compsep_run["ilc_bias"] == 0.:
        if "mask" in compsep_run:
            cov=np.mean(np.einsum('ik,jk->ijk', input_maps[:, compsep_run["mask"] > 0.], input_maps[:, compsep_run["mask"] > 0.]),axis=-1)
        else:
            cov=np.mean(np.einsum('ik,jk->ijk', input_maps, input_maps),axis=-1)
    else:
        if "mask" in compsep_run:
            cov = _get_local_cov(input_maps, lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=compsep_run["mask"], reduce_bias=compsep_run["reduce_ilc_bias"])
        else:                
            cov = _get_local_cov(input_maps, lmax, compsep_run["ilc_bias"], b_ell = b_ell, reduce_bias=compsep_run["reduce_ilc_bias"])
        if "mask" in compsep_run:
            if cov.shape[-1] == input_maps.shape[-1]:
                cov = np.copy(cov[...,compsep_run["mask"] > 0.])
    return cov

def get_inv_cov(cov):
    if cov.ndim == 2:
        inv_cov=np.linalg.inv(cov)
    elif cov.ndim == 3:
        inv_cov=np.linalg.inv(cov.T).T
    return inv_cov


