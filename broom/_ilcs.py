import numpy as np
import healpy as hp
from .configurations import Configs
from .routines import _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering,\
                      _get_local_cov, _EB_to_QU, _E_to_QU, _B_to_QU, _save_compsep_products,\
                      obj_to_array, array_to_obj, _get_full_path_out
from ._seds import _get_CMB_SED, _get_moments_SED, _standardize_cilc
from .clusters import _adapt_tracers_path, _cea_partition, _rp_partition, \
                      get_scalar_tracer, get_scalar_tracer_nl, initialize_scalar_tracers
from types import SimpleNamespace
import os


def ilc(config: Configs, input_alms, compsep_run, **kwargs):
    if compsep_run["method"] == "cilc" or compsep_run["method"] == "c_ilc":
        compsep_run = _standardize_cilc(compsep_run, config.lmax)

    if compsep_run["method"] in ["mcilc", "mc_ilc"]:
        if compsep_run["mc_type"] in ["cea_ideal", "rp_ideal"]:
            if not hasattr(input_alms, "fgds"):
                raise ValueError("The input_alms object must have fgds attribute for ideal tracer in MC-ILC.")
                    
    output_maps = _ilc(config, obj_to_array(input_alms), compsep_run, **kwargs)
    
    output_maps = _ilc_post_processing(config, output_maps, compsep_run, **kwargs)

    outputs = array_to_obj(output_maps, input_alms)

    del output_maps

    if config.save_compsep_products:
        _save_compsep_products(config, outputs, compsep_run, nsim=compsep_run["nsim"])
    if config.return_compsep_products:
        return outputs

def _ilc_post_processing(config: Configs, output_maps, compsep_run, **kwargs):
    if (output_maps.ndim==3) and ((output_maps.shape[0]==2 and config.field_out == "QU") or (output_maps.shape[0]==3 and config.field_out == "TQU")):
        outputs = np.zeros_like(output_maps)
        for c in range(output_maps.shape[-1]):
            outputs[:,:,c] = _EB_to_QU(output_maps[:,:,c],config.lmax, **kwargs)
        if "mask" in compsep_run:
            outputs[:,compsep_run["mask"] == 0.,:] = 0.
        return outputs
    elif (output_maps.ndim==2) and (config.field_out in ["QU_E", "QU_B"]):
        output = np.zeros((2, output_maps.shape[0], output_maps.shape[-1]))
        for c in range(output_maps.shape[-1]):
            if config.field_out == "QU_E":
                output[:,:,c] = _E_to_QU(output_maps[:,c],config.lmax, **kwargs)
            elif config.field_out=="QU_B":
                output[:,:,c] = _B_to_QU(output_maps[:,c],config.lmax, **kwargs)
        if "mask" in compsep_run:
            output[:,compsep_run["mask"] == 0.,:] = 0.
        return output
    else:
        if "mask" in compsep_run:
            if output_maps.ndim == 3:
                output_maps[:, compsep_run["mask"] == 0., :] = 0.
            elif output_maps.ndim == 2:
                output_maps[compsep_run["mask"] == 0.,:] = 0.
        return output_maps

def _ilc(config: Configs, input_alms, compsep_run, **kwargs):
    if input_alms.ndim == 4:
        if input_alms.shape[1] == 3:
            fields_ilc = ["T", "E", "B"]
        elif input_alms.shape[1] == 2:
            fields_ilc = ["E", "B"]
    elif input_alms.ndim == 3:
        if config.field_out in ["T", "E", "B"]:
            fields_ilc = [config.field_out]
        elif config.field_out in ["QU_E", "QU_B"]:
            fields_ilc = [config.field_out[-1]]

    if input_alms.ndim == 4:
        output_maps = np.zeros((input_alms.shape[1], 12 * config.nside**2, input_alms.shape[3]))
        for i in range(input_alms.shape[1]):
            compsep_run["field"] = fields_ilc[i]
            if compsep_run["method"] in ["mcilc", "mc_ilc"]:
                compsep_run["tracers"] = initialize_scalar_tracers(config, input_alms[:,i], compsep_run, field=compsep_run["field"], **kwargs)
            output_maps[i] = _ilc_scalar(config, input_alms[:, i, :, :], compsep_run, **kwargs)

    elif input_alms.ndim == 3:
        compsep_run["field"] = fields_ilc[0]
        if compsep_run["method"] in ["mcilc", "mc_ilc"]:
            compsep_run["tracers"] = initialize_scalar_tracers(config, input_alms, compsep_run, field=compsep_run["field"], **kwargs)
        output_maps = _ilc_scalar(config, input_alms, compsep_run, **kwargs)
    
    del compsep_run["field"]

    return output_maps

def _ilc_scalar(config: Configs, input_alms, compsep_run, **kwargs):
    if compsep_run["domain"] == "pixel":
        output_maps = _ilc_pixel(config, input_alms, compsep_run, **kwargs)
    elif compsep_run["domain"] == "needlet":
        output_maps = _ilc_needlet(config, input_alms, compsep_run, **kwargs)
    elif compsep_run["domain"] == "harmonic":
        output_maps = _hilc(config, input_alms, compsep_run)
    return output_maps

def _ilc_needlet(config: Configs, input_alms, compsep_run, **kwargs):
    b_ell = _get_needlet_windows_(compsep_run["needlet_config"], config.lmax)
    if compsep_run["b_squared"]:
        b_ell = b_ell**2

    if compsep_run['needlet_config']['save_needlets']:
        path_out = _get_full_path_out(config, compsep_run)
        os.makedirs(path_out, exist_ok=True)
        np.save(os.path.join(path_out, "needlet_bands"), b_ell)

    output_alms = np.zeros((input_alms.shape[1], input_alms.shape[-1]), dtype=complex)
    for j in range(b_ell.shape[0]):
        output_alms += _ilc_needlet_j(config, input_alms, compsep_run, b_ell[j], j, **kwargs)
    
    output_maps = np.array([hp.alm2map(np.ascontiguousarray(output_alms[:, c]), config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out) for c in range(input_alms.shape[-1])]).T
    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.,:] = 0.
    return output_maps
        
def _ilc_needlet_j(config: Configs, input_alms, compsep_run, b_ell, nl_scale, **kwargs):
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
#        if "mask" in compsep_run:
#            input_maps_nl[n] = np.array([(hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False) * compsep_run["mask"]) for c in range(input_alms.shape[-1])]).T            
#        else:
        input_maps_nl[n] = np.array([hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False) for c in range(input_alms.shape[-1])]).T

    if (compsep_run["method"]=="mcilc") or (compsep_run["method"]=="mc_ilc" and nl_scale in compsep_run["special_nls"]):
        tracer_nl = get_scalar_tracer_nl(compsep_run["tracers"], nside_, lmax_, b_ell)
        output_maps_nl = _mcilc_maps(config, input_maps_nl, tracer_nl, compsep_run, b_ell, nl_scale=nl_scale)
    else:
        output_maps_nl = _ilc_maps(config, input_maps_nl, compsep_run, b_ell, nl_scale=nl_scale)

    del input_maps_nl

    output_alms_nl = np.array([hp.map2alm(output_maps_nl[:,c], lmax=lmax_, pol=False, **kwargs) for c in range(output_maps_nl.shape[-1])]).T

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

def _ilc_pixel(config: Configs, input_alms, compsep_run, **kwargs):
    input_maps = np.zeros((input_alms.shape[0], 12 * config.nside**2, input_alms.shape[-1]))
    for n in range(input_alms.shape[0]):
#        if "mask" in compsep_run:
#            input_maps[n] = np.array([(hp.alm2map(np.ascontiguousarray(input_alms[n, :, c]), config.nside, lmax=config.lmax, pol=False) * compsep_run["mask"]) for c in range(input_alms.shape[-1])]).T
#        else:
        input_maps[n] = np.array([hp.alm2map(np.ascontiguousarray(input_alms[n, :, c]), config.nside, lmax=config.lmax, pol=False) for c in range(input_alms.shape[-1])]).T
    
    if compsep_run["method"]=="mcilc":
        tracer = get_scalar_tracer(compsep_run["tracers"])
        output_maps = _mcilc_maps(config, input_maps_nl, tracer, compsep_run, np.ones(config.lmax+1))    
    else:
        output_maps = _ilc_maps(config, input_maps, compsep_run, np.ones(config.lmax+1))
    
    if config.pixel_window_out:
        for c in range(output_maps.shape[1]):
            alm_out = hp.map2alm(output_maps[:,c],lmax=config.lmax, pol=False, **kwargs)
            output_maps[:,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=False, pixwin=True)
    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.,:] = 0.
    return output_maps

def _ilc_maps(config: Configs, input_maps, compsep_run, b_ell, nl_scale=None):
    good_channels_nl = _get_good_channels_nl(config, b_ell)
    
    if config.bandpass_integrate:
        if hasattr(config.instrument, "path_bandpasses"):
            bandwidths = []
            for i in good_channels_nl:
                bandwidths.append(os.path.join(config.instrument.path_bandpasses, f"bandpass_{config.instrument.channels_tags[i]}.npy"))  
        else: 
            bandwidths = np.array(config.instrument.bandwidth)[good_channels_nl]
    else:
        bandwidths = None
    A_cmb = _get_CMB_SED(np.array(config.instrument.frequency)[good_channels_nl], units=config.units)
        
    if compsep_run["method"] == "cilc":
        if nl_scale is None:
            compsep_run["A"] = _get_moments_SED(np.array(config.instrument.frequency)[good_channels_nl], compsep_run["constraints"]["moments"], beta_d=compsep_run["constraints"]["beta_d"], T_d=compsep_run["constraints"]["T_d"], beta_s=compsep_run["constraints"]["beta_s"], units=config.units, bandwidths=bandwidths)
            compsep_run["e"] = np.array(compsep_run["constraints"]["deprojection"])
        else:
            compsep_run["A"] = _get_moments_SED(np.array(config.instrument.frequency)[good_channels_nl], compsep_run["constraints"]["moments"][nl_scale], beta_d=compsep_run["constraints"]["beta_d"][nl_scale], T_d=compsep_run["constraints"]["T_d"][nl_scale], beta_s=compsep_run["constraints"]["beta_s"][nl_scale], units=config.units, bandwidths=bandwidths)
            compsep_run["e"] = np.array(compsep_run["constraints"]["deprojection"][nl_scale])

    elif (compsep_run["method"] == "c_ilc") and (nl_scale is not None) and (nl_scale in compsep_run["special_nls"]):
        compsep_run["A"] = _get_moments_SED(np.array(config.instrument.frequency)[good_channels_nl], compsep_run["constraints"]["moments"][compsep_run["special_nls"] == nl_scale], beta_d=compsep_run["constraints"]["beta_d"][compsep_run["special_nls"] == nl_scale], T_d=compsep_run["constraints"]["T_d"][compsep_run["special_nls"] == nl_scale], beta_s=compsep_run["constraints"]["beta_s"][compsep_run["special_nls"] == nl_scale], units=config.units, bandwidths=bandwidths)
        compsep_run["e"] = np.array(compsep_run["constraints"]["deprojection"][compsep_run["special_nls"] == nl_scale])

    cov = get_ilc_cov(input_maps[...,0], config.lmax, compsep_run, b_ell)
    inv_cov = get_inv_cov(cov)
    del cov

    w_ilc = get_ilc_weights(A_cmb, inv_cov, input_maps.shape, compsep_run)
    del inv_cov

    if compsep_run["save_weights"]:
        save_ilc_weights(config, w_ilc, compsep_run, hp.npix2nside(input_maps.shape[-2]), nl_scale=nl_scale)
    
    if "A" in compsep_run:
        del compsep_run['A'], compsep_run['e']

    if compsep_run["ilc_bias"] == 0.:
        output_maps = np.einsum('i,ijk->jk', w_ilc, input_maps)
    else:
        output_maps = np.einsum('ij,ijk->jk', w_ilc, input_maps)
    return output_maps

def _mcilc_maps(config: Configs, input_maps, tracer, compsep_run, b_ell, nl_scale=None):
    good_channels_nl = _get_good_channels_nl(config, b_ell)
    A_cmb = _get_CMB_SED(np.array(config.instrument.frequency)[good_channels_nl], units=config.units)

    if "cea" in compsep_run["mc_type"]:
        output_maps = _mcilc_cea_(config, input_maps, tracer, compsep_run, A_cmb, nl_scale=nl_scale)
    elif "rp" in compsep_run["mc_type"]:
        output_maps = _mcilc_rp_(config, input_maps, tracer, compsep_run, A_cmb, nl_scale=nl_scale)

    return output_maps


def _mcilc_cea_(config: Configs, input_maps, tracer, compsep_run, A_cmb, nl_scale=None):
    patches = _cea_partition(tracer, compsep_run["n_patches"])

    w_mcilc = get_mcilc_weights(input_maps[...,0], patches, A_cmb, compsep_run)
    if compsep_run["save_weights"]:
        save_ilc_weights(config, w_mcilc, compsep_run, hp.npix2nside(input_maps.shape[-2]), nl_scale=nl_scale)
    
    if "A" in compsep_run:
        del compsep_run['A'], compsep_run['e']

    output_maps = np.einsum('ij,ijk->jk', w_mcilc, input_maps)
    
    return output_maps

def _mcilc_rp_(config: Configs, input_maps, tracer, compsep_run, A_cmb, iterations=30, nl_scale=None):
    output_maps = np.zeros((input_maps.shape[1], input_maps.shape[-1]))

    for _ in range(iterations):    
        patches = _rp_partition(tracer, compsep_run["n_patches"])
        w_mcilc = get_mcilc_weights(input_maps[...,0], patches, A_cmb, compsep_run)
        if compsep_run["save_weights"]:
            if _ == 0:
                w_mcilc_save = np.copy(w_mcilc) / iterations
            else:
                w_mcilc_save += w_mcilc / iterations
        output_maps += (np.einsum('ij,ijk->jk', w_mcilc, input_maps) / iterations)

    if compsep_run["save_weights"]:
        save_ilc_weights(config, w_mcilc_save, compsep_run, hp.npix2nside(input_maps.shape[-2]), nl_scale=nl_scale)
    
    if "A" in compsep_run:
        del compsep_run['A'], compsep_run['e']
    
    return output_maps

def get_ilc_weights(A_cmb, inv_cov, input_shapes, compsep_run):
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
                    w_ilc[i, compsep_run["mask"] > 0.] = w_[i]
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
                    w_ilc[i, compsep_run["mask"] > 0.] = AT_invC[i]/AT_invC_A
                else:
                    w_ilc[i]=hp.ud_grade(AT_invC[i]/AT_invC_A,hp.npix2nside(input_shapes[-2]))
    return w_ilc

def get_mcilc_weights(inputs, patches, A_cmb, compsep_run):
    if "mask" in compsep_run:
        mask_mcilc = compsep_run["mask"]
    else:
        mask_mcilc = np.ones(inputs.shape[1])

    cov = get_mcilc_cov(inputs, patches, mask_mcilc, reduce_bias=compsep_run["reduce_mcilc_bias"])

    inv_cov = get_inv_cov(cov)
    del cov

    w_mcilc=np.zeros((inputs.shape[0],inputs.shape[1]))

    if "A" in compsep_run:
        compsep_run["A"] = np.vstack((A_cmb, compsep_run["A"]))
        compsep_run["e"] = np.insert(compsep_run["e"], 0, 1.)
        inv_ACA = np.linalg.inv((np.einsum("zi,ilk->zlk",compsep_run["A"],np.einsum("ijk,lj->ilk",inv_cov,compsep_run["A"]))).T).T
        w_mcilc[:, mask_mcilc > 0.] = (np.einsum("l,ljk->jk",compsep_run["e"],np.einsum("lzk,zjk->ljk",inv_ACA,np.einsum("zi,ijk->zjk",compsep_run["A"],inv_cov))))
        del w_, inv_ACA, inv_cov
    else:
        AT_invC = np.einsum('j,ijk->ik', A_cmb, inv_cov) # np.sum(inv_cov,axis=1)
        AT_invC_A = np.einsum('j,ijk, i->k', A_cmb, inv_cov, A_cmb) #np.sum(inv_cov,axis=(0,1))
        for i in range(inputs.shape[0]):
            w_mcilc[i, mask_mcilc > 0.] = AT_invC[i]/AT_invC_A
        del AT_invC, AT_invC_A, inv_cov
    
    return w_mcilc


def get_ilc_cov(input_maps, lmax, compsep_run, b_ell):
    if compsep_run["ilc_bias"] == 0.:
        if "mask" in compsep_run:
            cov=np.mean(np.einsum('ik,jk->ijk', (input_maps * compsep_run["mask"])[:, compsep_run["mask"] > 0.], (input_maps * compsep_run["mask"])[:, compsep_run["mask"] > 0.]),axis=-1)
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

def get_mcilc_cov(inputs, patches, mask_mcilc, reduce_bias=True, mcilc_rings=4):
    cov=np.zeros((inputs.shape[0], inputs.shape[0], np.sum(mask_mcilc > 0.)))
    
    if reduce_bias:
        neigh = hp.get_all_neighbours(hp.npix2nside(inputs.shape[1]), np.argwhere(mask_mcilc > 0.)[:, 0])
        for pix_ in (np.argwhere(mask_mcilc > 0.)[:, 0]):
            donut = np.ones(inputs.shape[1])
            donut[pix_] = 0.   
            if mcilc_rings > 0:
                donut[neigh[:,pix_]]=0.
            if mcilc_rings > 1:
                count=1
                neigh_=neigh[:,pix_]
                while count < mcilc_rings:
                    neigh_=neigh[:,neigh_].flatten()
                    donut[neigh_]=0.
                    count=count+1
            cov[...,pix_] = np.cov((inputs * mask_mcilc)[:,(patches==patches[pix_]) & (donut>0.) & (mask_mcilc>0.)],rowvar=True)
    else:
        for patch in np.unique(patches):
            cov[...,((patches==patch) & (mask_mcilc>0.))] = np.repeat(np.mean(np.einsum('ik,jk->ijk', (inputs * mask_mcilc)[:,((patches==patch) & (mask_mcilc>0.))], (inputs * mask_mcilc)[:,((patches==patch) & (mask_mcilc>0.))]),axis=2)[:, :, np.newaxis], np.sum(((patches==patch) & (mask_mcilc>0.))), axis=2)

    return cov

def get_inv_cov(cov):
    if cov.ndim == 2:
        inv_cov=np.linalg.inv(cov)
    elif cov.ndim == 3:
        inv_cov=np.linalg.inv(cov.T).T
    return inv_cov

def save_ilc_weights(config: Configs, w, compsep_run, nside_, nl_scale=None):
    path_out = _get_full_path_out(config, compsep_run)
    path_w = os.path.join(path_out, "weights")
    os.makedirs(path_w, exist_ok=True)
    filename = os.path.join(path_w, f"weights_{compsep_run['field']}_{config.fwhm_out}acm_ns{nside_}_lmax{config.lmax}")
    if nl_scale is not None:
        filename += f"_nl{nl_scale}"
    if compsep_run["nsim"] is not None:
        filename += f"_{compsep_run['nsim']}"
    np.save(filename, w)


