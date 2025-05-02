import numpy as np
import healpy as hp
from .configurations import Configs
from .routines import _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering,\
                      _get_local_cov, _EB_to_QU, _E_to_QU, _B_to_QU, _save_compsep_products,\
                      obj_to_array, array_to_obj, _get_full_path_out
from ._pilcs import get_pilc_cov, get_prilc_cov
from ._gilcs import _standardize_gnilc_run, Cn_C_Cn
from ._seds import _get_CMB_SED
import scipy
from numpy import linalg as lg
from types import SimpleNamespace
import os


def gpilc(config: Configs, input_alms, compsep_run, nsim=None):
    compsep_run = _standardize_gnilc_run(compsep_run, input_alms.total.shape[0], config.lmax)

    if hasattr(input_alms, "nuisance"):
        nuis_alms = getattr(input_alms, "nuisance")
    else:
        if config.verbose:
            print("No nuisance alms provided. Using input noise and CMB alms as nuisance.")
        if compsep_run["cmb_nuisance"]:
            nuis_alms = getattr(input_alms, "cmb") + getattr(input_alms, "noise")
        else:
            nuis_alms = getattr(input_alms, "noise")

    output_maps = _gpilc(config, obj_to_array(input_alms), nuis_alms, compsep_run)
    
    outputs = array_to_obj(output_maps, input_alms)

    del output_maps

    if config.save_compsep_products:
        _save_compsep_products(config, outputs, compsep_run, nsim=nsim)
    
    if config.return_compsep_products:
        return outputs

def _gpilc(config: Configs, input_alms, nuis_alms, compsep_run):
    if compsep_run["domain"] == "pixel":
        output_maps = _gpilc_pixel(config, input_alms, nuis_alms, compsep_run)
    elif compsep_run["domain"] == "needlet":
        output_maps = _gpilc_needlet(config, input_alms, nuis_alms, compsep_run)
    return output_maps

def _gpilc_pixel(config: Configs, input_alms, nuis_alms, compsep_run):
    input_maps = np.zeros((input_alms.shape[0], 2, 12 * config.nside**2, input_alms.shape[-1]))
    nuis_maps = np.zeros((input_alms.shape[0], 2, 12 * config.nside**2))

    for n in range(input_alms.shape[0]):
        if input_alms.ndim == 4:
            nuis_maps[n] = hp.alm2map(np.array([0.*nuis_alms[n, 0],nuis_alms[n, 0],nuis_alms[n, 1]]), config.nside, lmax=config.lmax, pol=True)[1:]      
        elif input_alms.ndim == 3:
            if config.field_out in ["QU_E", "E"]:
                nuis_maps[n] = hp.alm2map(np.array([0.*nuis_alms[n],nuis_alms[n],0.*nuis_alms[n]]), config.nside, lmax=config.lmax, pol=True)[1:]
            elif config.field_out in ["QU_B", "B"]:
                nuis_maps[n] = hp.alm2map(np.array([0.*nuis_alms[n],0.*nuis_alms[n],nuis_alms[n]]), config.nside, lmax=config.lmax, pol=True)[1:]

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
#            nuis_maps[n] = nuis_maps[n] * compsep_run["mask"]
#            for c in range(input_alms.shape[-1]):
#                input_maps[n,...,c] = input_maps[n,...,c] * compsep_run["mask"] 
                
    output_maps = _gpilc_maps(config, input_maps, nuis_maps, compsep_run, np.ones(config.lmax+1), depro_cmb=compsep_run["depro_cmb"], m_bias=compsep_run["m_bias"])

    if ((config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out) or (config.field_out in ["E","B"]) or (config.field_out=="EB"):
        for f, c in np.ndindex(output_maps.shape[0],output_maps.shape[-1]):
            if ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_purify" in config.leakage_correction):
                alm_out = purify_master(output_maps[f,...,c], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction))
                alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
            elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_recycling" in config.leakage_correction):
                if "_iterations" in config.leakage_correction:
                    iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
                else:
                    iterations = 0
                alm_out = purify_recycling(output_maps[f,...,c], output_maps[f,...,0], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction), iterations=iterations)
                alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
            else:
                alm_out = hp.map2alm([0.*output_maps[f,0,:,c],output_maps[f,0,:,c],output_maps[f,1,:,c]],lmax=config.lmax, pol=True)

            if (config.field_out in ["QU", "QU_E", "QU_B"]):
                output_maps[f,...,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=True, pixwin=config.pixel_window_out)[1:]
            elif config.field_out in ["E","B"]:
                output_maps[f,0,:,c] = hp.alm2map(alm_out[1] if config.field_out=="E" else alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
            elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and (config.leakage_correction=="mask_only"):
                alm_out = hp.map2alm(np.array([0.*output_maps[f,0,:,c],output_maps[f,0,:,c],output_maps[f,1,:,c]]) * compsep_run["mask"],lmax=config.lmax, pol=True)
            elif config.field_out=="EB":
                output_maps[f,0,:,c] = hp.alm2map(alm_out[1], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
                output_maps[f,1,:,c] = hp.alm2map(alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)

        if config.field_out in ["E","B"]:
            output_maps = output_maps[:,0]

    if "mask" in compsep_run:
        if output_maps.ndim == 3:
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        elif output_maps.ndim == 4:
            output_maps[...,compsep_run["mask"] == 0.,:] = 0.

    return output_maps

def _gpilc_needlet(config: Configs, input_alms, nuis_alms, compsep_run):
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    b_ell = b_ell**2

    if compsep_run['needlet_config']['save_needlets']:
        path_out = _get_full_path_out(config, compsep_run)
        os.makedirs(path_out, exist_ok=True)
        np.save(os.path.join(path_out, "needlet_bands"), b_ell)

    output_maps = np.zeros((len(compsep_run["channels_out"]), 2, hp.nside2npix(config.nside), input_alms.shape[-1]))
    
    for j in range(b_ell.shape[0]):
        output_maps += _gpilc_needlet_j(config, input_alms, nuis_alms, compsep_run, b_ell[j])
    
    if ((config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out) or (config.field_out in ["E","B"]) or (config.field_out=="EB"):
        for f, c in np.ndindex(output_maps.shape[0],output_maps.shape[-1]):
            if ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_purify" in config.leakage_correction):
                alm_out = purify_master(output_maps[f,...,c], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction))
                alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
            elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_recycling" in config.leakage_correction):
                if "_iterations" in config.leakage_correction:
                    iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
                else:
                    iterations = 0
                alm_out = purify_recycling(output_maps[f,...,c], output_maps[f,...,0], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction), iterations=iterations)
                alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
            elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and (config.leakage_correction=="mask_only"):
                alm_out = hp.map2alm(np.array([0.*output_maps[f,0,:,c],output_maps[f,0,:,c],output_maps[f,1,:,c]]) * compsep_run["mask"],lmax=config.lmax, pol=True)
            else:
                alm_out = hp.map2alm([0.*output_maps[f,0,:,c],output_maps[f,0,:,c],output_maps[f,1,:,c]],lmax=config.lmax, pol=True)

            if (config.field_out in ["QU", "QU_E", "QU_B"]):
                output_maps[f,...,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=True, pixwin=config.pixel_window_out)[1:]
            elif config.field_out in ["E","B"]:
                output_maps[f,0,:,c] = hp.alm2map(alm_out[1] if config.field_out=="E" else alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
            elif config.field_out=="EB":
                output_maps[f,0,:,c] = hp.alm2map(alm_out[1], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
                output_maps[f,1,:,c] = hp.alm2map(alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)

        if config.field_out in ["E","B"]:
            output_maps = output_maps[:,0]

    if "mask" in compsep_run:
        if output_maps.ndim == 3:
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        elif output_maps.ndim == 4:
            output_maps[...,compsep_run["mask"] == 0.,:] = 0.
    
    return output_maps
        
def _gpilc_needlet_j(config: Configs, input_alms, nuis_alms, compsep_run, b_ell, depro_cmb=None, m_bias=0):
    if "mask" in compsep_run:
        nside_, lmax_ = config.nside, config.lmax
    else:
        if compsep_run["adapt_nside"]:
            nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
        else:
            nside_, lmax_ = config.nside, config.lmax

#    good_channels_nl = _get_good_channels_nl(config, b_ell)
    good_channels_nl = np.arange(input_alms.shape[0])

    input_maps_nl = np.zeros((good_channels_nl.shape[0], 2, 12 * nside_**2, input_alms.shape[-1]))
    nuis_maps_nl = np.zeros((good_channels_nl.shape[0], 2, 12 * nside_**2))

    for n, channel in enumerate(good_channels_nl):
        input_alms_j = np.zeros((2, hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)
        nuis_alms_j = np.zeros((2, hp.Alm.getsize(lmax_)), dtype=complex)
        if input_alms.ndim == 4:
            for k in range(2):
                input_alms_j[k] = _needlet_filtering(input_alms[channel,k], b_ell, lmax_)
                nuis_alms_j[k] = _needlet_filtering(nuis_alms[channel,k], b_ell, lmax_)
        elif input_alms.ndim == 3:
            if config.field_out in ["QU_E", "E"]:
                input_alms_j[0] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
                nuis_alms_j[0] = _needlet_filtering(nuis_alms[channel], b_ell, lmax_)
            elif config.field_out in ["QU_B", "B"]:
                input_alms_j[1] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
                nuis_alms_j[1] = _needlet_filtering(nuis_alms[channel], b_ell, lmax_)
        
        input_alms_j = np.ascontiguousarray(input_alms_j)
        nuis_alms_j = np.ascontiguousarray(nuis_alms_j)

        if ("mask" in compsep_run) and (config.mask_type == "observed_patch"):
            nuis_maps_nl[n] = (hp.alm2map(np.array([0.*nuis_alms_j[0],nuis_alms_j[0],nuis_alms_j[1]]), nside_, lmax=lmax_, pol=True)[1:]) * compsep_run["mask"]       
            for c in range(input_alms.shape[-1]):
                input_maps_nl[n,...,c] = (hp.alm2map(np.array([0.*input_alms_j[0, :, c],input_alms_j[0, :, c],input_alms_j[1, :, c]]), nside_, lmax=lmax_, pol=True)[1:]) * compsep_run["mask"]       
        else:
            nuis_maps_nl[n] = hp.alm2map(np.array([0.*nuis_alms_j[0],nuis_alms_j[0],nuis_alms_j[1]]), nside_, lmax=lmax_, pol=True)[1:]   
            for c in range(input_alms.shape[-1]):
                input_maps_nl[n,...,c] = hp.alm2map(np.array([0.*input_alms_j[0, :, c],input_alms_j[0, :, c],input_alms_j[1, :, c]]), nside_, lmax=lmax_, pol=True)[1:]   

    output_maps_nl = _gpilc_maps(config, input_maps_nl, nuis_maps_nl, compsep_run, b_ell, depro_cmb=depro_cmb, m_bias=m_bias)
    del input_maps_nl, nuis_maps_nl

    if input_alms.ndim == 4:
        output_alms_j = np.zeros((input_alms.shape[0], 2, hp.Alm.getsize(config.lmax), input_alms.shape[-1]), dtype=complex)
    elif input_alms.ndim == 3:
        output_alms_j = np.zeros((input_alms.shape[0], hp.Alm.getsize(config.lmax), input_alms.shape[-1]), dtype=complex)

    for n, channel in enumerate(good_channels_nl):
        if input_alms.ndim == 4:
            output_alms_nl = np.zeros((2, hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)
        elif input_alms.ndim == 3:
            output_alms_nl = np.zeros((hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)

        for c in range(input_alms.shape[-1]):
            if ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_purify" in config.leakage_correction):
                alm_out = purify_master(output_maps_nl[n,...,c], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction))
            elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_recycling" in config.leakage_correction):
                if "_iterations" in config.leakage_correction:
                    iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
                else:
                    iterations = 0
                alm_out = purify_recycling(output_maps_nl[n,...,c], output_maps_nl[n,...,0], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction), iterations=iterations)
            else:
                alm_out = hp.map2alm(np.concatenate([(0.*output_maps_nl[n,0,:,c])[np.newaxis],output_maps_nl[n,...,c]], axis=0), lmax=lmax_, pol=True)[1:]

            if input_alms.ndim == 4:
                output_alms_nl[...,c] = alm_out
            elif input_alms.ndim == 3:
                if config.field_out in ["QU_E", "E"]:
                    output_alms_nl[...,c] = alm_out[0]
                elif config.field_out in ["QU_B", "B"]:
                    output_alms_nl[...,c] = alm_out[1]

        if input_alms.ndim == 4:
            for k in range(2):
                if compsep_run["b_squared"]:
                    output_alms_j[channel,k] = _needlet_filtering(output_alms_nl[k], np.ones(lmax_+1), config.lmax)
                else:
                    output_alms_j[channel,k] = _needlet_filtering(output_alms_nl[k], b_ell[:lmax_+1], config.lmax)
        elif input_alms.ndim == 3:
            if compsep_run["b_squared"]:
                output_alms_j[channel] = _needlet_filtering(output_alms_nl, np.ones(lmax_+1), config.lmax)
            else:
                output_alms_j[channel] = _needlet_filtering(output_alms_nl, b_ell[:lmax_+1], config.lmax)

    return output_alms_j

def _gpilc_maps(config: Configs, input_maps, nuis_maps, compsep_run, b_ell, depro_cmb=None, m_bias=0):

    cov = (get_prilc_cov(input_maps[...,0], config.lmax, compsep_run, b_ell)).T
    cov_n = (get_prilc_cov(nuis_maps, config.lmax, compsep_run, b_ell)).T

    λ, U = Cn_C_Cn(cov,cov_n)
    λ[λ<1.]=1.

    W = _get_gpilc_weights(config, U, λ, cov, cov_n, input_maps.shape, compsep_run, depro_cmb=depro_cmb, m_bias=m_bias) 

    del cov, cov_n, U, λ

    if compsep_run["ilc_bias"] == 0.:
        if W.ndim==2:
            output_maps = np.einsum('li,ifjk->lfjk', W, input_maps)
        elif W.ndim==3:
            output_maps = np.zeros((W.shape[0],2,input_maps.shape[-2],input_maps.shape[-1]))
            output_maps[:,0] = np.einsum('li,ijk->ljk', W[0], input_maps[:,0]) - np.einsum('li,ijk->ljk', W[1], input_maps[:,1])
            output_maps[:,1] = np.einsum('li,ijk->ljk', W[1], input_maps[:,0]) + np.einsum('li,ijk->ljk', W[0], input_maps[:,1])
    else:
        if W.ndim==3:
            output_maps = np.einsum('jli,ifjk->lfjk', W, input_maps)
        elif W.ndim==4:
            output_maps = np.zeros((W.shape[0],2,input_maps.shape[-2],input_maps.shape[-1]))
            output_maps[:,0] = np.einsum('jli,ijk->ljk', W[0], input_maps[:,0]) - np.einsum('jli,ijk->ljk', W[1], input_maps[:,1])
            output_maps[:,1] = np.einsum('jli,ijk->ljk', W[1], input_maps[:,0]) + np.einsum('jli,ijk->ljk', W[0], input_maps[:,1])

    return output_maps[compsep_run["channels_out"]]

def _get_gpilc_weights(config: Configs, U, λ, cov, cov_n, input_shapes, compsep_run, depro_cmb=None, m_bias=0):

    if cov.ndim == 2:
        inv_cov = lg.inv(cov)

        A_m=np.zeros(λ.shape[0]+1)
        for i in range(λ.shape[0]):
            A_m[i]=2*i + np.sum((λ[i:]-np.log(λ[i:])-1.))
        A_m[λ.shape[0]]=2.*λ.shape[0]

        m = np.argmin(A_m)

        if m_bias != 0:
            m = m + int(m_bias)  

        U_s = np.delete(U,np.where(λ < λ[m-1]),axis=1)
        λ_s = np.delete(λ,np.where(λ < λ[m-1]))

        F = lg.multi_dot([scipy.linalg.sqrtm(cov_n),U_s])
        
        if depro_cmb is None:
            W = lg.multi_dot([F,lg.inv(lg.multi_dot([F.T,inv_cov,F])),F.T,inv_cov])
        else:
            A_cmb = _get_CMB_SED(np.array(config.instrument.frequency), units=config.units)

            F_e = np.column_stack([F, depro_cmb * np.ones(input_shapes[0])])
            F_A = np.column_stack([F, A_cmb])
            W = lg.multi_dot([F_e,lg.inv(lg.multi_dot([F_A.T,inv_cov,F_A])),F_A.T,inv_cov])   

    elif cov.ndim == 3:
        A_m=np.zeros((λ.shape[0],λ.shape[1]+1))
        for i in range(λ.shape[1]):
            A_m[:,i]=2*i + np.sum((λ[:,i:]-np.log(λ[:,i:])-1.),axis=1)

        A_m[:,λ.shape[1]]=2.*λ.shape[1]
        m = np.argmin(A_m,axis=1)
        
        if m_bias != 0:
            m = m + int(m_bias)  

        covn_sqrt = lg.cholesky(cov_n)
        covn_sqrt_inv = lg.inv(covn_sqrt)
        A_cmb = _get_CMB_SED(np.array(config.instrument.frequency), units=config.units)

        W_=np.zeros((cov.shape[0],input_shapes[0],input_shapes[0]))

        for m_ in np.unique(m):
            U_s=U[m==m_,:,:m_]
            cov_inv = lg.inv(cov[m==m_])
            F = np.einsum("kij,kjz->kiz", covn_sqrt[m==m_],U_s)

            if depro_cmb is None: 
                W_[m==m_] = np.einsum("kil,klj->kij",F,np.einsum("kzl,klj->kzj",lg.inv(np.einsum("kiz,kij,kjl->kzl",F,cov_inv,F)),np.einsum("kiz,kij->kzj",F,cov_inv)))
            else:
                e_cmb = depro_cmb * np.ones((F.shape[0],F.shape[1],1)) 
                F_e = np.concatenate((F,e_cmb),axis=2)
                F_A = np.concatenate((F,np.tile(A_cmb, (F.shape[0], 1))[:, :, np.newaxis]),axis=2)
                W_[m==m_] = np.einsum("kil,klj->kij",F_e,np.einsum("kzl,klj->kzj",lg.inv(np.einsum("kiz,kij,kjl->kzl",F_A,cov_inv,F_A)),np.einsum("kiz,kij->kzj",F_A,cov_inv)))

        if ("mask" not in compsep_run) and (W_.shape[0] != input_shapes[-2]):
            W = np.zeros((input_shapes[-2],W_.shape[1],W_.shape[2]))
            for i, k in np.ndindex(W_.shape[1],W_.shape[2]):
                W[:,i,k]=hp.ud_grade(W_[:,i,k],hp.npix2nside(input_shapes[-2]))
        else:
            W=np.copy(W_)

    return W

def _gpilc_needlet_old(config: Configs, input_alms, nuis_alms, compsep_run):
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    if compsep_run["b_squared"]:
        b_ell = b_ell**2

    if compsep_run['needlet_config']['save_needlets']:
        path_out = _get_full_path_out(config, compsep_run)
        os.makedirs(path_out, exist_ok=True)
        np.save(os.path.join(path_out, "needlet_bands"), b_ell)

    if input_alms.ndim == 4:
        output_alms = np.zeros((len(compsep_run["channels_out"]), 2, input_alms.shape[-2], input_alms.shape[-1]), dtype=complex)
    elif input_alms.ndim == 3:
        output_alms = np.zeros((len(compsep_run["channels_out"]), input_alms.shape[-2], input_alms.shape[-1]), dtype=complex)

    for j in range(b_ell.shape[0]):
        output_alms += _gpilc_needlet_j_old(config, input_alms, nuis_alms, compsep_run, b_ell[j])
    
    output_alms = np.ascontiguousarray(output_alms)

    if config.field_out in ["QU", "QU_E", "QU_B", "EB"]:
        output_maps = np.zeros((output_alms.shape[0], 2, 12 * config.nside**2, input_alms.shape[-1]))
    else:
        output_maps = np.zeros((output_alms.shape[0], 12 * config.nside**2, input_alms.shape[-1]))

    for f, c in np.ndindex(output_alms.shape[0],output_alms.shape[-1]):
        if config.field_out=="QU":
            output_maps[f,...,c] = hp.alm2map(np.concatenate([(0.*output_alms[f,0,:,c])[np.newaxis],output_alms[f,...,c]], axis=0), config.nside, lmax=config.lmax, pol=True, pixwin=config.pixel_window_out)[1:]
        elif config.field_out=="QU_E":
            output_maps[f,...,c] = hp.alm2map(np.array([0.*output_alms[f,:,c],output_alms[f,:,c],0.*output_alms[f,:,c]]), config.nside, lmax=config.lmax, pol=True, pixwin=config.pixel_window_out)[1:]
        elif config.field_out=="QU_B":
            output_maps[f,...,c] = hp.alm2map(np.array([0.*output_alms[f,:,c],0.*output_alms[f,:,c],output_alms[f,:,c]]), config.nside, lmax=config.lmax, pol=True, pixwin=config.pixel_window_out)[1:]
        elif config.field_out=="EB":
            output_maps[f,...,c] = np.array([hp.alm2map(np.ascontiguousarray(output_alms[f,j,:,c]), config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out) for j in range(2)])
        else:
            output_maps[f,:,c] = hp.alm2map(np.ascontiguousarray(output_alms[f,:,c]), config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
       
    if "mask" in compsep_run:
        if output_maps.ndim == 3:
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        elif output_maps.ndim == 4:
            output_maps[...,compsep_run["mask"] == 0.,:] = 0.

    return output_maps
        
def _gpilc_needlet_j_old(config: Configs, input_alms, nuis_alms, compsep_run, b_ell, depro_cmb=None, m_bias=0):
    if "mask" in compsep_run:
        nside_, lmax_ = config.nside, config.lmax
    else:
        if compsep_run["adapt_nside"]:
            nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
        else:
            nside_, lmax_ = config.nside, config.lmax

#    good_channels_nl = _get_good_channels_nl(config, b_ell)
    good_channels_nl = np.arange(input_alms.shape[0])

    input_maps_nl = np.zeros((good_channels_nl.shape[0], 2, 12 * nside_**2, input_alms.shape[-1]))
    nuis_maps_nl = np.zeros((good_channels_nl.shape[0], 2, 12 * nside_**2))

    for n, channel in enumerate(good_channels_nl):
        input_alms_j = np.zeros((2, hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)
        nuis_alms_j = np.zeros((2, hp.Alm.getsize(lmax_)), dtype=complex)
        if input_alms.ndim == 4:
            for k in range(2):
                input_alms_j[k] = _needlet_filtering(input_alms[channel,k], b_ell, lmax_)
                nuis_alms_j[k] = _needlet_filtering(nuis_alms[channel,k], b_ell, lmax_)
        elif input_alms.ndim == 3:
            if config.field_out in ["QU_E", "E"]:
                input_alms_j[0] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
                nuis_alms_j[0] = _needlet_filtering(nuis_alms[channel], b_ell, lmax_)
            elif config.field_out in ["QU_B", "B"]:
                input_alms_j[1] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
                nuis_alms_j[1] = _needlet_filtering(nuis_alms[channel], b_ell, lmax_)
        
        input_alms_j = np.ascontiguousarray(input_alms_j)
        nuis_alms_j = np.ascontiguousarray(nuis_alms_j)

#        if ("mask" in compsep_run) and (config.mask_type == "observed_patch"):
#            nuis_maps_nl[n] = (hp.alm2map(np.array([0.*nuis_alms_j[0],nuis_alms_j[0],nuis_alms_j[1]]), nside_, lmax=lmax_, pol=True)[1:]) * compsep_run["mask"]       
#            for c in range(input_alms.shape[-1]):
#                input_maps_nl[n,...,c] = (hp.alm2map(np.array([0.*input_alms_j[0, :, c],input_alms_j[0, :, c],input_alms_j[1, :, c]]), nside_, lmax=lmax_, pol=True)[1:]) * compsep_run["mask"]       
#        else:
        nuis_maps_nl[n] = hp.alm2map(np.array([0.*nuis_alms_j[0],nuis_alms_j[0],nuis_alms_j[1]]), nside_, lmax=lmax_, pol=True)[1:]   
        for c in range(input_alms.shape[-1]):
            input_maps_nl[n,...,c] = hp.alm2map(np.array([0.*input_alms_j[0, :, c],input_alms_j[0, :, c],input_alms_j[1, :, c]]), nside_, lmax=lmax_, pol=True)[1:]   

    output_maps_nl = _gpilc_maps(config, input_maps_nl, nuis_maps_nl, compsep_run, b_ell, depro_cmb=depro_cmb, m_bias=m_bias)
    del input_maps_nl, nuis_maps_nl

    if input_alms.ndim == 4:
        output_alms_j = np.zeros((input_alms.shape[0], 2, hp.Alm.getsize(config.lmax), input_alms.shape[-1]), dtype=complex)
    elif input_alms.ndim == 3:
        output_alms_j = np.zeros((input_alms.shape[0], hp.Alm.getsize(config.lmax), input_alms.shape[-1]), dtype=complex)

    for n, channel in enumerate(good_channels_nl):
        if input_alms.ndim == 4:
            output_alms_nl = np.zeros((2, hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)
        elif input_alms.ndim == 3:
            output_alms_nl = np.zeros((hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)

        for c in range(input_alms.shape[-1]):
            if ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_purify" in config.leakage_correction):
                alm_out = purify_master(output_maps_nl[n,...,c], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction))
            elif ("mask" in compsep_run) and (config.mask_type == "observed_patch") and ("_recycling" in config.leakage_correction):
                if "_iterations" in config.leakage_correction:
                    iterations = int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
                else:
                    iterations = 0
                alm_out = purify_recycling(output_maps_nl[n,...,c], output_maps_nl[n,...,0], compsep_run["mask"], config.lmax, purify_E=("E" in config.leakage_correction), iterations=iterations)
            else:
                alm_out = hp.map2alm(np.concatenate([(0.*output_maps_nl[n,0,:,c])[np.newaxis],output_maps_nl[n,...,c]], axis=0), lmax=lmax_, pol=True)[1:]

            if input_alms.ndim == 4:
                output_alms_nl[...,c] = alm_out
            elif input_alms.ndim == 3:
                if config.field_out in ["QU_E", "E"]:
                    output_alms_nl[...,c] = alm_out[0]
                elif config.field_out in ["QU_B", "B"]:
                    output_alms_nl[...,c] = alm_out[1]

        if input_alms.ndim == 4:
            for k in range(2):
                if compsep_run["b_squared"]:
                    output_alms_j[channel,k] = _needlet_filtering(output_alms_nl[k], np.ones(lmax_+1), config.lmax)
                else:
                    output_alms_j[channel,k] = _needlet_filtering(output_alms_nl[k], b_ell[:lmax_+1], config.lmax)
        elif input_alms.ndim == 3:
            if compsep_run["b_squared"]:
                output_alms_j[channel] = _needlet_filtering(output_alms_nl, np.ones(lmax_+1), config.lmax)
            else:
                output_alms_j[channel] = _needlet_filtering(output_alms_nl, b_ell[:lmax_+1], config.lmax)

    return output_alms_j
