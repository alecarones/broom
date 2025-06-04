import numpy as np
import healpy as hp
from .configurations import Configs
from .routines import _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering,\
                      _get_local_cov, _EB_to_QU, _E_to_QU, _B_to_QU, _save_compsep_products,\
                      obj_to_array, array_to_obj, _get_full_path_out
from ._ilcs import _standardize_cilc, _get_good_channels_nl, get_inv_cov
from .leakage import purify_master, purify_recycling
from ._seds import _get_CMB_SED, _get_moments_SED, _standardize_cilc
from types import SimpleNamespace
import os
import re


def pilc(config: Configs, input_alms, compsep_run, **kwargs):
    if config.field_out not in ["E", "B", "QU", "EB", "QU_E", "QU_B"]:
        raise ValueError("Invalid field_out for PILC. It must be E, B, QU, EB, QU_E, or QU_B.")

    if compsep_run["method"] == "cpilc" or compsep_run["method"] == "c_pilc":
        compsep_run = _standardize_cilc(compsep_run, config.lmax)

    output_maps = _pilc(config, obj_to_array(input_alms), compsep_run, **kwargs)
    
    outputs = array_to_obj(output_maps, input_alms)

    del output_maps

    if config.save_compsep_products:
        _save_compsep_products(config, outputs, compsep_run, nsim=compsep_run["nsim"])
    if config.return_compsep_products:
        return outputs

def _pilc(config: Configs, input_alms, compsep_run, **kwargs):
    if input_alms.ndim == 4:
        if input_alms.shape[1] != 2:
            raise ValueError("input_alms must have shape (nfreq, 2, nalm, ncomps) for pilc.")
        compsep_run["field"] = "QU"

    elif input_alms.ndim == 3:
        if config.field_out in ["E", "QU_E"]:
            compsep_run["field"] = "QU_E"
        elif config.field_out in ["B", "QU_B"]:
            compsep_run["field"] = "QU_B"

    if compsep_run["domain"] == "pixel":
        output_maps = _pilc_pixel(config, input_alms, compsep_run, **kwargs)
    elif compsep_run["domain"] == "needlet":
        output_maps = _pilc_needlet(config, input_alms, compsep_run, **kwargs)

    del compsep_run["field"]

    return output_maps

def _pilc_needlet(config: Configs, input_alms, compsep_run, **kwargs):
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    b_ell = b_ell**2

    if compsep_run['needlet_config']['save_needlets']:
        path_out = _get_full_path_out(config, compsep_run)
        os.makedirs(path_out, exist_ok=True)
        np.save(os.path.join(path_out, "needlet_bands"), b_ell)
    
    output_maps = np.zeros((2, hp.nside2npix(config.nside), input_alms.shape[-1]))
    
    for j in range(b_ell.shape[0]):
        output_maps += _pilc_needlet_j(config, input_alms, compsep_run, b_ell[j], j)
    
    if ((config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out) or (config.field_out in ["E","B"]) or (config.field_out=="EB"):
        for c in range(output_maps.shape[-1]):
            if ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_purify" in config.leakage_correction):
                alm_out = purify_master(output_maps[...,c], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction))
                alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
            elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_recycling" in config.leakage_correction):
                if "_iterations" in config.leakage_correction:
                    iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
                else:
                    iterations = 0
                alm_out = purify_recycling(output_maps[...,c], output_maps[...,0], np.ceil(compsep_run["mask"]), config.lmax, purify_E=("E" in config.leakage_correction), iterations=iterations)
                alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
            elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and (config.leakage_correction=="mask_only"):
                alm_out = hp.map2alm(np.array([0.*output_maps[0,:,c],output_maps[0,:,c],output_maps[1,:,c]]) * compsep_run["mask"],lmax=config.lmax, pol=True, **kwargs)
            else:
                alm_out = hp.map2alm([0.*output_maps[0,:,c],output_maps[0,:,c],output_maps[1,:,c]],lmax=config.lmax, pol=True, **kwargs)

            if (config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out:
                output_maps[...,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=True, pixwin=True)[1:]
            elif config.field_out in ["E","B"]:
                output_maps[0,:,c] = hp.alm2map(alm_out[1] if config.field_out=="E" else alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
            elif config.field_out=="EB":
                output_maps[0,:,c] = hp.alm2map(alm_out[1], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
                output_maps[1,:,c] = hp.alm2map(alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)

        if config.field_out in ["E","B"]:
            output_maps = output_maps[0]

    if "mask" in compsep_run:
        if output_maps.ndim == 3:
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        elif output_maps.ndim == 2:
            output_maps[compsep_run["mask"] == 0.,:] = 0.

    return output_maps

def _pilc_needlet_j(config: Configs, input_alms, compsep_run, b_ell, nl_scale):
    
    nside_, lmax_ = config.nside, config.lmax

    good_channels_nl = _get_good_channels_nl(config, b_ell)

    input_maps_nl = np.zeros((good_channels_nl.shape[0], 2, 12 * nside_**2, input_alms.shape[-1]))
    for n, channel in enumerate(good_channels_nl):
        input_alms_j = np.zeros((2, hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)
        if input_alms.ndim == 4:
            for k in range(2):
                input_alms_j[k] = _needlet_filtering(input_alms[channel,k], b_ell, lmax_)
        elif input_alms.ndim == 3:
            if config.field_out in ["QU_E", "E"]:
                input_alms_j[0] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
            elif config.field_out in ["QU_B", "B"]:
                input_alms_j[1] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        
        input_alms_j = np.ascontiguousarray(input_alms_j)

#        if ("mask" in compsep_run) and (config.mask_type == "observed_patch"):
#            for c in range(input_alms.shape[-1]):
#                input_maps_nl[n,...,c] = (hp.alm2map(np.array([0.*input_alms_j[0, :, c],input_alms_j[0, :, c],input_alms_j[1, :, c]]), nside_, lmax=lmax_, pol=True)[1:]) * compsep_run["mask"]       
#        else:
        for c in range(input_alms.shape[-1]):
            input_maps_nl[n,...,c] = hp.alm2map(np.array([0.*input_alms_j[0, :, c],input_alms_j[0, :, c],input_alms_j[1, :, c]]), nside_, lmax=lmax_, pol=True)[1:]   

    output_maps_nl = _pilc_maps(config, input_maps_nl, compsep_run, b_ell, nl_scale=nl_scale)
    
    del input_maps_nl

    return output_maps_nl

def _pilc_pixel(config: Configs, input_alms, compsep_run, **kwargs):
    
    input_maps = np.zeros((input_alms.shape[0], 2, 12 * config.nside**2, input_alms.shape[-1]))
    for n in range(input_alms.shape[0]):
        for c in range(input_alms.shape[-1]):
            if input_alms.ndim == 4:
                input_maps[n,...,c] = hp.alm2map(np.array([0.*input_alms[n, 0, :, c],input_alms[n, 0, :, c],input_alms[n, 1, :, c]]), config.nside, lmax=config.lmax, pol=True)[1:]      
            elif input_alms.ndim == 3:
                if config.field_out in ["QU_E", "E"]:
                    input_maps[n,...,c] = hp.alm2map(np.array([0.*input_alms[n, :, c],input_alms[n, :, c],0.*input_alms[n, :, c]]), config.nside, lmax=config.lmax, pol=True)[1:]
                elif config.field_out in ["QU_B", "B"]:
                    input_maps[n,...,c] = hp.alm2map(np.array([0.*input_alms[n, :, c],0.*input_alms[n, :, c],input_alms[n, :, c]]), config.nside, lmax=config.lmax, pol=True)[1:]

#    if ("mask" in compsep_run) and (config.mask_type == "observed_patch"):
#        for n in range(input_alms.shape[0]):
#            for c in range(input_alms.shape[-1]):
#                input_maps[n,...,c] = input_maps[n,...,c] * compsep_run["mask"] 

    output_maps = _pilc_maps(config, input_maps, compsep_run, np.ones(config.lmax+1))

    if ((config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out) or (config.field_out in ["E","B"]) or (config.field_out=="EB"):
        for c in range(output_maps.shape[-1]):
            if ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_purify" in config.leakage_correction):
                alm_out = purify_master(output_maps[...,c], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction))
                alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
            elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_recycling" in config.leakage_correction):
                if "_iterations" in config.leakage_correction:
                    iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
                else:
                    iterations = 0
                alm_out = purify_recycling(output_maps[...,c], output_maps[...,0], np.ceil(compsep_run["mask"]), config.lmax, purify_E=("E" in config.leakage_correction), iterations=iterations)
                alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
            elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and (config.leakage_correction=="mask_only"):
                alm_out = hp.map2alm(np.array([0.*output_maps[0,:,c],output_maps[0,:,c],output_maps[1,:,c]]) * compsep_run["mask"],lmax=config.lmax, pol=True, **kwargs)
            else:
                alm_out = hp.map2alm([0.*output_maps[0,:,c],output_maps[0,:,c],output_maps[1,:,c]],lmax=config.lmax, pol=True, **kwargs)

            if (config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out:
                output_maps[...,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=True, pixwin=True)[1:]
            elif config.field_out in ["E","B"]:
                output_maps[0,:,c] = hp.alm2map(alm_out[1] if config.field_out=="E" else alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
            elif config.field_out=="EB":
                output_maps[0,:,c] = hp.alm2map(alm_out[1], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
                output_maps[1,:,c] = hp.alm2map(alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)

        if config.field_out in ["E","B"]:
            output_maps = output_maps[0]

    if "mask" in compsep_run:
        if output_maps.ndim == 3:
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        elif output_maps.ndim == 2:
            output_maps[compsep_run["mask"] == 0.,:] = 0.

    return output_maps
   
def _pilc_maps(config: Configs, input_maps, compsep_run, b_ell, nl_scale=None):
    good_channels_nl = _get_good_channels_nl(config, b_ell)
    if config.bandpass_integrate:
        if hasattr(config.instrument, "path_bandpasses"):
            bandwidths = config.instrument.path_bandpasses
        else:
            bandwidths = np.array(config.instrument.bandwidth)[good_channels_nl]
    else:
        bandwidths = None
    A_cmb = _get_CMB_SED(np.array(config.instrument.frequency)[good_channels_nl], units=config.units)
        
    if compsep_run["method"] == "cpilc":
        if nl_scale is None:
            compsep_run["A"] = _get_moments_SED(np.array(config.instrument.frequency)[good_channels_nl], compsep_run["constraints"]["moments"], beta_d=compsep_run["constraints"]["beta_d"], T_d=compsep_run["constraints"]["T_d"], beta_s=compsep_run["constraints"]["beta_s"], units=config.units, bandwidths=bandwidths)
            compsep_run["e"] = np.array(compsep_run["constraints"]["deprojection"])
        else:
            compsep_run["A"] = _get_moments_SED(np.array(config.instrument.frequency)[good_channels_nl], compsep_run["constraints"]["moments"][nl_scale], beta_d=compsep_run["constraints"]["beta_d"][nl_scale], T_d=compsep_run["constraints"]["T_d"][nl_scale], beta_s=compsep_run["constraints"]["beta_s"][nl_scale], units=config.units, bandwidths=bandwidths)
            compsep_run["e"] = np.array(compsep_run["constraints"]["deprojection"][nl_scale])

    elif (compsep_run["method"] == "c_pilc") and (nl_scale is not None) and (nl_scale in compsep_run["special_nls"]):
        compsep_run["A"] = _get_moments_SED(np.array(config.instrument.frequency)[good_channels_nl], compsep_run["constraints"]["moments"][compsep_run["special_nls"] == nl_scale], beta_d=compsep_run["constraints"]["beta_d"][compsep_run["special_nls"] == nl_scale], T_d=compsep_run["constraints"]["T_d"][compsep_run["special_nls"] == nl_scale], beta_s=compsep_run["constraints"]["beta_s"][compsep_run["special_nls"] == nl_scale], units=config.units, bandwidths=bandwidths)
        compsep_run["e"] = np.array(compsep_run["constraints"]["deprojection"][compsep_run["special_nls"] == nl_scale])

    cov = get_prilc_cov(input_maps[...,0], config.lmax, compsep_run, b_ell)
    inv_cov = get_inv_cov(cov)
    del cov

    w_pilc = get_pilc_weights(A_cmb, inv_cov, input_maps.shape, compsep_run)
    del inv_cov

    if compsep_run["save_weights"]:
        path_out = _get_full_path_out(config, compsep_run)
        path_w = os.path.join(path_out, "weights")
        os.makedirs(path_w, exist_ok=True)
        filename = os.path.join(path_w, f"weights_{compsep_run['field']}_{config.fwhm_out}acm_ns{input_maps.shape[-2]}_lmax{config.lmax}")
        if nl_scale is not None:
            filename += f"_nl{nl_scale}"
        if compsep_run["nsim"] is not None:
            filename += f"_{compsep_run['nsim']}"
        np.save(filename, w_pilc)
    
    if "A" in compsep_run:
        del compsep_run['A'], compsep_run['e']

    if compsep_run["ilc_bias"] == 0.:
        if w_pilc.ndim==1:
            output_maps = np.einsum('i,ifjk->fjk', w_pilc, input_maps)
        elif w_pilc.ndim==2:
            output_maps = []
            output_maps.append(np.einsum('i,ijk->jk', w_pilc[0], input_maps[:,0]) - np.einsum('i,ijk->jk', w_pilc[1], input_maps[:,1]))
            output_maps.append(np.einsum('i,ijk->jk', w_pilc[1], input_maps[:,0]) + np.einsum('i,ijk->jk', w_pilc[0], input_maps[:,1]))
            output_maps = np.array(output_maps)
    else:
        if w_pilc.ndim==2:
            output_maps = np.einsum('ij,ifjk->fjk', w_pilc, input_maps)
        elif w_pilc.ndim==3:
            output_maps = []
            output_maps.append(np.einsum('ij,ijk->jk', w_pilc[0], input_maps[:,0]) - np.einsum('ij,ijk->jk', w_pilc[1], input_maps[:,1]))
            output_maps.append(np.einsum('ij,ijk->jk', w_pilc[1], input_maps[:,0]) + np.einsum('ij,ijk->jk', w_pilc[0], input_maps[:,1]))
            output_maps = np.array(output_maps)

    return output_maps

def get_pilc_weights(A_cmb, inv_cov, input_shapes, compsep_run):
    if "A" in compsep_run:
        compsep_run["A"] = np.vstack((A_cmb, compsep_run["A"]))
        compsep_run["e"] = np.insert(compsep_run["e"], 0, 1.)
        if compsep_run["ilc_bias"] == 0.:
            inv_ACA = np.linalg.inv((np.einsum("zi,il->zl",compsep_run["A"],np.einsum("ij,lj->il",inv_cov,compsep_run["A"]))))
            w_ilc = (np.einsum("l,lj->j",compsep_run["e"],np.einsum("lz,zj->lj",inv_ACA,np.einsum("zi,ij->zj",compsep_run["A"],inv_cov))))
            del inv_ACA
        else:
            inv_ACA = np.linalg.inv((np.einsum("zi,ilk->zlk",compsep_run["A"],np.einsum("ijk,lj->ilk",inv_cov,compsep_run["A"]))).T).T
            w_ilc=np.zeros((input_shapes[0],input_shapes[-2]))
            w_ = (np.einsum("l,ljk->jk",compsep_run["e"],np.einsum("lzk,zjk->ljk",inv_ACA,np.einsum("zi,ijk->zjk",compsep_run["A"],inv_cov))))
            for i in range(input_shapes[0]):
                if "mask" in compsep_run:
                    w_ilc[i] = w_[i]
                else:
                    w_ilc[i]=hp.ud_grade(w_[i],hp.npix2nside(input_shapes[-2]))
            del w_, inv_ACA
    else:
        if compsep_run["ilc_bias"] == 0.:
    #        w_ilc = np.sum(inv_cov,axis=1)/np.sum(inv_cov)
            w_ilc = (A_cmb.T @ inv_cov) / (A_cmb.T @ inv_cov @ A_cmb) 
        else:
            w_ilc=np.zeros((input_shapes[0],input_shapes[-2]))
            AT_invC = np.einsum('j,ijk->ik', A_cmb, inv_cov) # np.sum(inv_cov,axis=1)
            AT_invC_A = np.einsum('j,ijk, i->k', A_cmb, inv_cov, A_cmb) #np.sum(inv_cov,axis=(0,1))
            for i in range(input_shapes[0]):
                if "mask" in compsep_run:
                    w_ilc[i] = AT_invC[i]/AT_invC_A
                else:
                    w_ilc[i]=hp.ud_grade(AT_invC[i]/AT_invC_A,hp.npix2nside(input_shapes[-2]))
    return w_ilc


def get_pilc_cov(input_maps, lmax, compsep_run, b_ell):
    cov = []

    if compsep_run["ilc_bias"] == 0.:
        if "mask" in compsep_run:
            cov.append(np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0, compsep_run["mask"] > 0.], input_maps[:, 0, compsep_run["mask"] > 0.]),axis=-1) + np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1, compsep_run["mask"] > 0.], input_maps[:, 1, compsep_run["mask"] > 0.]),axis=-1))
            cov.append(np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0, compsep_run["mask"] > 0.], input_maps[:, 1, compsep_run["mask"] > 0.]),axis=-1) - np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1, compsep_run["mask"] > 0.], input_maps[:, 0, compsep_run["mask"] > 0.]),axis=-1))
        else:
            cov.append(np.mean(np.einsum('ik,jk->ijk', input_maps[:,0], input_maps[:,0]),axis=-1) + np.mean(np.einsum('ik,jk->ijk', input_maps[:,1], input_maps[:,1]),axis=-1))
            cov.append(np.mean(np.einsum('ik,jk->ijk', input_maps[:,0], input_maps[:,1]),axis=-1) - np.mean(np.einsum('ik,jk->ijk', input_maps[:,1], input_maps[:,0]),axis=-1))
        cov = np.array(cov)
    else:
        if "mask" in compsep_run:
            cov.append(_get_local_cov(input_maps[:,0], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=compsep_run["mask"], reduce_bias=compsep_run["reduce_ilc_bias"]) + _get_local_cov(input_maps[:,1], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=compsep_run["mask"], reduce_bias=compsep_run["reduce_ilc_bias"]))
            cov.append(_get_local_cov(input_maps[:,0], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=compsep_run["mask"], reduce_bias=compsep_run["reduce_ilc_bias"], input_maps_2=input_maps[:,1]) - _get_local_cov(input_maps[:,1], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=compsep_run["mask"], reduce_bias=compsep_run["reduce_ilc_bias"], input_maps_2=input_maps[:,0]))
        else:                
            cov.append(_get_local_cov(input_maps[:,0], lmax, compsep_run["ilc_bias"], b_ell = b_ell, reduce_bias=compsep_run["reduce_ilc_bias"]) + _get_local_cov(input_maps[:,1], lmax, compsep_run["ilc_bias"], b_ell = b_ell, reduce_bias=compsep_run["reduce_ilc_bias"]))
            cov.append(_get_local_cov(input_maps[:,0], lmax, compsep_run["ilc_bias"], b_ell = b_ell, reduce_bias=compsep_run["reduce_ilc_bias"], input_maps_2=input_maps[:,1]) - _get_local_cov(input_maps[:,1], lmax, compsep_run["ilc_bias"], b_ell = b_ell, reduce_bias=compsep_run["reduce_ilc_bias"],input_maps_2=input_maps[:,0]))
        cov = np.array(cov)
        if "mask" in compsep_run:
            if cov.shape[-1] == input_maps.shape[-1]:
                cov[...,compsep_run["mask"] == 0.] = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0, compsep_run["mask"] > 0.], input_maps[:, 0, compsep_run["mask"] > 0.]),axis=-1) + np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1, compsep_run["mask"] > 0.], input_maps[:, 1, compsep_run["mask"] > 0.]),axis=-1)

    return np.concatenate((np.concatenate((cov[0],-cov[1]),axis=1),np.concatenate((cov[1],cov[0]),axis=1)),axis=0)

def get_prilc_cov(input_maps, lmax, compsep_run, b_ell):
    if compsep_run["ilc_bias"] == 0.:
        if "mask" in compsep_run:
            cov = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0, compsep_run["mask"] > 0.], input_maps[:, 0, compsep_run["mask"] > 0.]),axis=-1) + np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1, compsep_run["mask"] > 0.], input_maps[:, 1, compsep_run["mask"] > 0.]),axis=-1)
        else:
            cov = np.mean(np.einsum('ik,jk->ijk', input_maps[:,0], input_maps[:,0]),axis=-1) + np.mean(np.einsum('ik,jk->ijk', input_maps[:,1], input_maps[:,1]),axis=-1)
    else:
        if "mask" in compsep_run:
            cov = _get_local_cov(input_maps[:,0], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=compsep_run["mask"], reduce_bias=compsep_run["reduce_ilc_bias"]) + _get_local_cov(input_maps[:,1], lmax, compsep_run["ilc_bias"], b_ell = b_ell, mask=compsep_run["mask"], reduce_bias=compsep_run["reduce_ilc_bias"])
        else:                
            cov = _get_local_cov(input_maps[:,0], lmax, compsep_run["ilc_bias"], b_ell = b_ell, reduce_bias=compsep_run["reduce_ilc_bias"]) + _get_local_cov(input_maps[:,1], lmax, compsep_run["ilc_bias"], b_ell = b_ell, reduce_bias=compsep_run["reduce_ilc_bias"])
        if "mask" in compsep_run:
            if cov.shape[-1] == input_maps.shape[-1]:
                cov[...,compsep_run["mask"] == 0.] = np.mean(np.einsum('ik,jk->ijk', input_maps[:, 0, compsep_run["mask"] > 0.], input_maps[:, 0, compsep_run["mask"] > 0.]),axis=-1) + np.mean(np.einsum('ik,jk->ijk', input_maps[:, 1, compsep_run["mask"] > 0.], input_maps[:, 1, compsep_run["mask"] > 0.]),axis=-1)

    return cov

def _pilc_needlet_old(config: Configs, input_alms, compsep_run):
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    if compsep_run["b_squared"]:
        b_ell = b_ell**2

    if compsep_run['needlet_config']['save_needlets']:
        path_out = _get_full_path_out(config, compsep_run)
        os.makedirs(path_out, exist_ok=True)
        np.save(os.path.join(path_out, "needlet_bands"), b_ell)
    
    output_alms = np.zeros((2, input_alms.shape[-2], input_alms.shape[-1]), dtype=complex)
    
    for j in range(b_ell.shape[0]):
        output_alms += _pilc_needlet_j_old(config, input_alms, compsep_run, b_ell[j], j)
    
    if config.field_out in ["QU", "QU_E", "QU_B", "EB"]:
        output_maps = np.zeros((2, 12 * config.nside**2, input_alms.shape[-1]))
    else:
        output_maps = np.zeros((12 * config.nside**2, input_alms.shape[-1]))

    for c in range(input_alms.shape[-1]):
        if config.field_out=="QU":
            output_maps[...,c] = hp.alm2map(np.concatenate([(0.*output_alms[0,:,c])[np.newaxis],output_alms[...,c]], axis=0), config.nside, lmax=config.lmax, pol=True, pixwin=config.pixel_window_out)[1:]
        elif config.field_out=="QU_E":
            output_maps[...,c] = hp.alm2map(np.array([0.*output_alms[:,c],output_alms[:,c],0.*output_alms[:,c]]), config.nside, lmax=config.lmax, pol=True, pixwin=config.pixel_window_out)[1:]
        elif config.field_out=="QU_B":
            output_maps[...,c] = hp.alm2map(np.array([0.*output_alms[:,c],0.*output_alms[:,c],output_alms[:,c]]), config.nside, lmax=config.lmax, pol=True, pixwin=config.pixel_window_out)[1:]
        elif config.field_out=="EB":
            output_maps[...,c] = np.array([hp.alm2map(np.ascontiguousarray(output_alms[j,:,c]), config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out) for j in range(2)])
        else:
            output_maps[:,c] = hp.alm2map(np.ascontiguousarray(output_alms[:,c]), config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
    
    if "mask" in compsep_run:
        if output_maps.ndim == 3:
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        elif output_maps.ndim == 2:
            output_maps[compsep_run["mask"] == 0.,:] = 0.

    return output_maps

def _pilc_needlet_j_old(config: Configs, input_alms, compsep_run, b_ell, nl_scale, **kwargs):
    
    if "mask" in compsep_run:
        nside_, lmax_ = config.nside, config.lmax
    else:
        if compsep_run["adapt_nside"]:
            nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
        else:
            nside_, lmax_ = config.nside, config.lmax

    good_channels_nl = _get_good_channels_nl(config, b_ell)

    input_maps_nl = np.zeros((good_channels_nl.shape[0], 2, 12 * nside_**2, input_alms.shape[-1]))
    for n, channel in enumerate(good_channels_nl):
        input_alms_j = np.zeros((2, hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)
        if input_alms.ndim == 4:
            for k in range(2):
                input_alms_j[k] = _needlet_filtering(input_alms[channel,k], b_ell, lmax_)
        elif input_alms.ndim == 3:
            if config.field_out in ["QU_E", "E"]:
                input_alms_j[0] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
            elif config.field_out in ["QU_B", "B"]:
                input_alms_j[1] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        
        input_alms_j = np.ascontiguousarray(input_alms_j)

#        if ("mask" in compsep_run) and (config.mask_type == "observed_patch"):
#            for c in range(input_alms.shape[-1]):
#                input_maps_nl[n,...,c] = (hp.alm2map(np.array([0.*input_alms_j[0, :, c],input_alms_j[0, :, c],input_alms_j[1, :, c]]), nside_, lmax=lmax_, pol=True)[1:]) * compsep_run["mask"]       
#        else:
        for c in range(input_alms.shape[-1]):
            input_maps_nl[n,...,c] = hp.alm2map(np.array([0.*input_alms_j[0, :, c],input_alms_j[0, :, c],input_alms_j[1, :, c]]), nside_, lmax=lmax_, pol=True)[1:]   

    output_maps_nl = _pilc_maps(config, input_maps_nl, compsep_run, b_ell, nl_scale=nl_scale)
    
    del input_maps_nl

    if input_alms.ndim == 4:
        output_alms_nl = np.zeros((2, hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)
    elif input_alms.ndim == 3:
        output_alms_nl = np.zeros((hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)

    for c in range(input_alms.shape[-1]):
        if ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_purify" in config.leakage_correction):
            alm_out = purify_master(output_maps_nl[...,c], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction))
        elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_recycling" in config.leakage_correction):
            if "_iterations" in config.leakage_correction:
                iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
            else:
                iterations = 0
            alm_out = purify_recycling(output_maps_nl[...,c], output_maps_nl[...,0], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction), iterations=iterations)
        elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and (config.leakage_correction == "mask_only"):
            alm_out = hp.map2alm(np.concatenate([(0.*output_maps_nl[0,:,c])[np.newaxis],output_maps_nl[...,c]], axis=0) * compsep_run["mask"], lmax=lmax_, pol=True, **kwargs)[1:]
        else:
            alm_out = hp.map2alm(np.concatenate([(0.*output_maps_nl[0,:,c])[np.newaxis],output_maps_nl[...,c]], axis=0), lmax=lmax_, pol=True, **kwargs)[1:]

        if input_alms.ndim == 4:
            output_alms_nl[...,c] = alm_out
        elif input_alms.ndim == 3:
            if config.field_out in ["QU_E", "E"]:
                output_alms_nl[...,c] = alm_out[0]
            elif config.field_out in ["QU_B", "B"]:
                output_alms_nl[...,c] = alm_out[1]

    if input_alms.ndim == 4:
        output_alms_j = np.zeros((2, hp.Alm.getsize(config.lmax), input_alms.shape[-1]), dtype=complex)
        for k in range(2):
            if compsep_run["b_squared"]:
                output_alms_j[k] = _needlet_filtering(output_alms_nl[k], np.ones(lmax_+1), config.lmax)
            else:
                output_alms_j[k] = _needlet_filtering(output_alms_nl[k], b_ell[:lmax_+1], config.lmax)
    elif input_alms.ndim == 3:
        if compsep_run["b_squared"]:
            output_alms_j = _needlet_filtering(output_alms_nl, np.ones(lmax_+1), config.lmax)
        else:
            output_alms_j = _needlet_filtering(output_alms_nl, b_ell[:lmax_+1], config.lmax)

    return output_alms_j
