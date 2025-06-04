import numpy as np 
import healpy as hp
import os
from .routines import _needlet_filtering, _slice_data, _map2alm_kwargs
from .configurations import Configs
from .simulations import _get_data_foregrounds_, _get_data_simulations_

def get_and_save_real_tracers_B(config: Configs, systematics=None, **kwargs):
    """
    Get the realistic tracers.
    """
    from broom import component_separation

    config_mc = get_mc_config(config)
    kwargs = _map2alm_kwargs(**kwargs)

    if config.verbose:
        print("Generating input simulations for MC-ILC tracers")
    mc_data = get_mc_data(config_mc)
    if systematics is not None:
        print("Adding systematic effect to the data")
        systematics = _adapt_systamatics(systematics, mc_data.total.shape, config.lmax, data_type=config_mc.data_type, **kwargs)
        mc_data.total = mc_data.total + systematics
    config_mc.compsep = get_tracers_compsep(config.real_mc_tracers[0]["channels_tracers"], config.lmax)

    if config.verbose:
        print(f"Generating the MC-ILC tracers for {config.experiment} experiment and {''.join(config.foreground_models)} foreground model")
    tracers = component_separation(config_mc, mc_data)
    
    tracers = _combine_B_tracers(tracers.total)

    if config.verbose:
        print(f"Saving the tracers in {config.real_mc_tracers[0]['path_tracers']} directory")
    _save_real_tracers_B(tracers, config.real_mc_tracers[0]["path_tracers"], np.array(config_mc.instrument.channels_tags)[config.real_mc_tracers[0]["channels_tracers"]], config_mc.fwhm_out, config_mc.lmax)

def _adapt_systematics(systematics, data_shape, lmax, data_type="maps", pixel_window_in=False, **kwargs):
    """
    Combine the data with the systematics.
    """

    if systematics.shape[0] != data_shape[0]:
        raise ValueError("The systematics and data must have the same number of channels.")

    if data_type == "maps":
        if (systematics.ndim != 3) or (systematics.shape[1] not in [2, 3]) or (systematics.shape[-1] < data_shape[-1]):
            raise ValueError("If 'data_type' is 'maps', the systematics must be a 3D array with shape (n_channels, n_fields, n_pixels) with n_fields = 2 or 3 and nside larger or equal to the chosen nside.")
        else:
            if systematics.shape[-1] > data_shape[-1]:
                if systematics.shape[1] == 2:
                    systematics = np.concatenate((np.zeros((systematics.shape[0], 1, systematics.shape[-1])), systematics), axis=1)
                systematics_ = np.zeros((systematics.shape[0], 2, data_shape[-1]))
                for i in range(systematics.shape[0]):
                    alm_syst = hp.map2alm(systematics[i], lmax=lmax, pol=True, **kwargs)
                    if pixel_window_in:
                        pw_in = hp.pixwin(hp.npix2nside(systematics.shape[-1]), lmax=lmax, pol=True)
                        pw_out = hp.pixwin(hp.npix2nside(data_shape[-1]), lmax=lmax, pol=True)
                        pw = pw_out / pw_in
                        for k in range(1,3):
                            alm_syst[k] = hp.almxfl(alm_syst[k], pw[1])
                    systematics_[i] = hp.alm2map(alm_syst, nside=hp.npix2nside(data_shape[-1]), lmax=lmax, pol=True)[1:]
                return systematics_              
            else:
                if systematics.shape[1] == 2:
                    return systematics
                elif systematics.shape[1] == 3:
                    return systematics[:, 1:]
    elif data_type == "alms":
        if (systematics.ndim == 3 and systematics.shape[1] not in [1, 2, 3]) or (systematics.shape[-1] < data_shape[-1]):
            raise ValueError("If 'data_type' is 'alms', the systematics must be a 2D or 3D array with shape (n_channels, ..., n_alms) with lmax larger or equal to the chosen lmax.")
        else:
            if hp.Alm.getlmax(data_shape[-1]) < hp.Alm.getlmax(systematics.shape[-1]):
                idx_lmax = np.array([hp.Alm.getidx(hp.Alm.getlmax(systematics.shape[-1]), ell, m) for ell in range(2, hp.Alm.getlmax(data_shape[-1]) + 1) for m in range(ell + 1)])
                idx_config_lmax = np.array([hp.Alm.getidx(hp.Alm.getlmax(data_shape[-1]), ell, m) for ell in range(2, hp.Alm.getlmax(data_shape[-1]) + 1) for m in range(ell + 1)])
                systematics_ = np.zeros((systematics.shape[0], data_shape[-1]), dtype=complex)
                if systematics.ndim==2:
                    systematics_[:, idx_config_lmax] = systematics[:, idx_lmax]
                elif systematics.ndim==3:
                    systematics_[:, idx_config_lmax] = systematics[:, -1, idx_lmax]
                return systematics_
            else:
                if systematics.ndim == 2:
                    return systematics
                elif systematics.ndim == 3:
                    return systematics[:, -1]

def _save_real_tracers_B(tracers, path_tracers, tags, fwhm_out, lmax):
    """
    Save the tracers to the given path.
    """
    if not os.path.exists(path_tracers):
        os.makedirs(path_tracers)

    if not path_tracers.endswith('/'):
        path_tracers = path_tracers + '/'

    for i, tracer in enumerate(tracers):
        hp.write_map(path_tracers + f"B_tracer_{tags[i]}_{fwhm_out}acm_ns{hp.npix2nside(tracer.shape[0])}_lmax{lmax}.fits", tracer, overwrite=True)
    
def initialize_scalar_tracers(config: Configs, input_alms, compsep_run, field="B", **kwargs):
    if compsep_run["mc_type"] in ["cea_ideal", "rp_ideal"]:
        if compsep_run["domain"] == "needlet":
            tracers = np.array([input_alms[compsep_run["channels_tracers"][0],:,1],input_alms[compsep_run["channels_tracers"][1],:,1]])
        elif compsep_run["domain"] == "pixel":
            tracers = np.array([hp.alm2map(input_alms[compsep_run["channels_tracers"][0],:,1],config.nside,lmax=config.lmax,pol=False),hp.alm2map(input_alms[compsep_run["channels_tracers"][1],:,1],config.nside,lmax=config.lmax,pol=False)])
    if compsep_run["mc_type"] in ["cea_real", "rp_real"]:
        tracers_tags = []
        for freq_tracer in compsep_run["channels_tracers"]:
            tracers_tags.append(config.instrument.channels_tags[freq_tracer])
        if compsep_run["domain"] == "needlet":
            tracers = load_scalar_tracers_for_ratio(get_tracers_paths_for_ratio(config, compsep_run["path_tracers"], tracers_tags, field=field), config.nside, config.lmax, return_alms=True, **kwargs)
        elif compsep_run["domain"] == "pixel":
            tracers = load_scalar_tracers_for_ratio(get_tracers_paths_for_ratio(config, compsep_run["path_tracers"], tracers_tags, field=field), config.nside, config.lmax, return_alms=False, **kwargs)
    return tracers

def get_tracers_paths_for_ratio(config: Configs, path_tracers, tracers_tags, field="B"):
    tracers_paths = []
    for tracer_tag in tracers_tags:
        tracers_paths.append(path_tracers + f"{field}_tracer_{tracer_tag}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}.fits")
    missing_tracers = []
    for n, tracer_path in enumerate(tracers_paths):
        if not os.path.exists(tracer_path):
            missing_tracers.append(tracers_tags[n])
    if len(missing_tracers) > 0:
        raise ValueError(f"Missing tracer files: {missing_tracers}. Please check the paths or run the tracer generation routine.") 
    else:
        return tracers_paths

def load_scalar_tracers_for_ratio(tracers_paths, nside, lmax, field="B", return_alms=True, **kwargs):
    tracers = []
    for tracer_path in tracers_paths:
        tracer = hp.read_map(tracer_path, field=0)
        if return_alms:
            tracer = hp.map2alm(tracer, lmax=lmax, pol=False, **kwargs)
        else:
            if hp.get_nside(tracer) != nside:
                alm_ = hp.map2alm(tracer, lmax=lmax, pol=False, **kwargs)
                tracer = hp.alm2map(alm_, nside, lmax=lmax, pol=False)
        tracers.append(tracer)

    return np.array(tracers)     

def get_scalar_tracer_nl(tracers, nside_, lmax_, b_ell):
    if tracers.ndim == 1:
        alm_tracer_nl = _needlet_filtering(tracers, b_ell, lmax_)
        return hp.alm2map(alm_tracer_nl, nside_, lmax=lmax_, pol=False)
    elif tracers.ndim == 2:
        tracers_nl = []
        for tracer in tracers:
            alm_tracer_nl = _needlet_filtering(tracer, b_ell, lmax_)
            tracers_nl.append(hp.alm2map(alm_tracer_nl, nside_, lmax=lmax_, pol=False))
        tracers_nl = np.array(tracers_nl)
        return tracers_nl[0] / tracers_nl[1]

def get_scalar_tracer(tracers):
    if tracers.ndim == 1:
        return tracers
    elif tracers.ndim == 2:
        return tracers[0] / tracers[1]

def _cea_partition(map_,n_patches):
    split = np.array(np.array_split(np.sort(map_),n_patches))
    patches = np.zeros(12 * (hp.get_nside(map_))**2)
    
    for n in range(n_patches):
        if n==0:
            patches[map_ <= np.max(split[n])] = 1. * n
        elif n==(n_patches-1):
            patches[map_ >= np.min(split[n])] = 1. * n
        else:
            patches[(np.min(split[n]) <= map_) & (map_ <= np.max(split[n]))] = 1. * n
            
    return patches

def _rp_partition(map_,n_patches):

    min_fraction = 1.25 / n_patches

    partition = np.zeros(n_patches)
    while (np.any(partition < min_fraction)):
        partition = np.random.uniform(low=0.0, high=1.0, size=n_patches)
        partition = partition / np.sum(partition)
    partition = np.cumsum(partition)
    
    bins = [0]
    for i in np.arange(1,n_patches):
        bins.append(int((map_.shape[0]) * partition[i-1]))
    bins.append(map_.shape[0])

    patches = np.zeros(12 * (hp.get_nside(map_))**2)
    for i in range(n_patches):
        patches[np.argsort(map_)[bins[i]:bins[i+1]]]=i

    return patches

def _adapt_tracers_path(path_tracers, n_fields = 1):
    if n_fields == 1:
        if isinstance(path_tracers, str):
            return path_tracers
        elif isinstance(path_tracers, list):
            if len(path_tracers) == 1:
                return path_tracers[0]
            else:
                raise ValueError("If path_tracers is a list, it must contain only one element if you want to analyse just one field.")
    else:
        if isinstance(path_tracers, str):
            return [path_tracers] * n_fields
        elif isinstance(path_tracers, list):
            if len(path_tracers) == n_fields:
                return path_tracers
            else:
                raise ValueError("If path_tracers is a list, it must contain as many elements as the number of fields to be analysed.")

def get_tracers_compsep(channels_tracers, lmax):   
    #merging_needlets_in = [0,16,20,25,34]
    #merging_needlets = [0,16]
    #needlet_config = {}
    #needlet_config["needlet_windows"] = "mexican"
    #needlet_config["width"] = 1.3
    #bl = _get_needlet_windows_(needlet_config, lmax)

    tracers_compsep = [
    {"method": "gilc",
    "domain": "needlet",
    "ilc_bias": 0.13,
    "needlet_config":[
    {"needlet_windows": "mexican",
    "width": 1.3,
    "merging_needlets": [0,16,20,25,34]}],
    "channels_out": channels_tracers,
    },
    {"method": "gilc",
    "domain": "needlet",
    "ilc_bias": 0.13,
    "needlet_config":[
    {"needlet_windows": "mexican",
    "width": 1.3,
    "merging_needlets": [0,12,15,20,25,34]}],
    "channels_out": channels_tracers,
    }]
    return tracers_compsep

def get_mc_config(config: Configs):
    config_mc = Configs(config=config.to_dict_for_mc())
    config_mc.return_fgd_components = False
    if config_mc.bandpass_integrate:
        config_mc.data_path = f"inputs_mc_tracers/{config_mc.experiment}/total/total_bp_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
        config_mc.fgds_path = f"inputs_mc_tracers/{config_mc.experiment}/foregrounds/{''.join(config_mc.foreground_models)}/foregrounds_bp_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
    else:
        config_mc.data_path = f"inputs_mc_tracers/{config_mc.experiment}/total/total_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
        config_mc.fgds_path = f"inputs_mc_tracers/{config_mc.experiment}/foregrounds/{''.join(config_mc.foreground_models)}/foregrounds_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
    config_mc.noise_path = f"inputs_mc_tracers/{config_mc.experiment}/noise/noise_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
    config_mc.cmb_path = f"inputs_mc_tracers/{config_mc.experiment}/cmb/cmb_{config_mc.data_type}_ns{config_mc.nside}_lmax{config_mc.lmax}"
    if not os.path.exists(config_mc.data_path + ".npy") or not os.path.exists(config_mc.noise_path + ".npy") or not os.path.exists(config_mc.cmb_path + ".npy"):
        config_mc.generate_input_foregrounds = True #
        config_mc.generate_input_noise = True #
        config_mc.generate_input_cmb = True #
        config_mc.cls_cmb_path = "utils/Cls_Planck2018_lensed_r0.fits"
        config_mc.generate_input_data = True #
        config_mc.save_inputs = True # 
        config_mc.seed_cmb = None
        config_mc.seed_noise = None
    else:
        if not os.path.exists(config_mc.fgds_path + f"_{''.join(config_mc.foreground_models)}.npy"):
            config_mc.generate_input_foregrounds = True
            config_mc.generate_input_data = True #
            config_mc.save_inputs = True
        else:
            config_mc.generate_input_foregrounds = False
            config_mc.generate_input_data = False
            config_mc.save_inputs = False #
        config_mc.generate_input_noise = False #
        config_mc.generate_input_cmb = False #

    config_mc.save_compsep_products = False #
    config_mc.return_compsep_products = True
    config_mc.mask_type = "mask_for_compsep"
    config_mc.verbose = False

    if config_mc.data_type == "maps":
        config_mc.field_in = "QU"
        config_mc.mc_data_field = "TQU"
    elif config_mc.data_type == "alms":
        config_mc.field_in = "B"
        config_mc.mc_data_field = "TEB"
    config_mc.field_out = "B"

    return config_mc

def get_mc_data(config_mc: Configs):
    mc_foregrounds = _get_data_foregrounds_(config_mc)
    mc_data = _get_data_simulations_(config_mc, mc_foregrounds)
    return _slice_data(mc_data, config_mc.mc_data_field, config_mc.field_in)

def _combine_B_tracers(tracers, coefficients=[0.7,0.3]):
    """
    Combine the tracers using the given coefficients.
    """
    if tracers.ndim == 2:
        return tracers
    elif tracers.ndim == 3:
        if len(coefficients) != tracers.shape[0]:
            raise ValueError("The number of coefficients must match the number of tracers.")
        coefficients = np.array(coefficients) / np.sum(coefficients)
        return np.einsum("i,ijk->jk", coefficients, tracers)


