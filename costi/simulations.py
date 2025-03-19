import numpy as np 
import healpy as hp
import pysm3.units as u
import os
import pysm3
from astropy.io import fits
from .configurations import Configs
from .routines import _get_ell_filter, _get_beam_from_file

def _get_full_simulations(config: Configs):
    if config.nsim_start is not None:
        if not isinstance(config.nsim_start, int):
            raise ValueError("nsims_start must be an integer.")

    foregrounds = _get_data_foregrounds_(config)

    data, cmb, noise = _get_data_simulations_(config, foregrounds, nsim=nsim)

def _get_data_foregrounds_(config: Configs):
    if config.generate_input_simulations:
        if config.input_beams == "guassian":
            foregrounds = _get_foregrounds(config.foreground_models, config.instrument, config.nside, return_components=config.return_fgd_components, pixel_window=config.pixel_window_in, units=config.units, return_alms=(config.data_type=="alms"), bandpass_integrate=config.bandpass_integrate, lmin=config.lmin)
        else:
            bl_path = os.path.join(config.inputs_path, "beams", config.experiment)
            foregrounds = _get_foregrounds(config.foreground_models, config.instrument, config.nside, return_components=config.return_fgd_components, pixel_window=config.pixel_window_in, units=config.units, return_alms=(config.data_type=="alms"), bandpass_integrate=config.bandpass_integrate, lmin=config.lmin, bl_path=bl_path, symmetric_beam=config.input_beams=="file_lm")
        if config.save_input_simulations:
            _save_input_foregrounds(config, foregrounds)
    elif config.load_input_simulations:
        if config.return_fgd_components:
            if config.data_type == "alms":
                foregrounds = np.zeros((len(config.instrument.frequency), 3, hp.Alm.getsize(3*config.nside-1), len(config.foreground_models)+1))
            else:
                foregrounds = np.zeros((len(config.instrument.frequency), 3, hp.nside2npix(config.nside), len(config.foreground_models)+1))
            for idx, fmodel in enumerate(config.foreground_models):
                foregrounds[..., idx + 1] = _load_input_foregrounds(config.fgds_path + f"/foregrounds_{fmodel}_{config.data_type}_ns{config.nside}" + (f"_lmin{config.lmin}" if config.lmin > 2 else ""))
        else:
            if config.data_type == "alms":
                foregrounds = np.zeros((len(config.instrument.frequency), 3, hp.Alm.getsize(3*config.nside-1),1), dtype=complex)
            else:
                foregrounds = np.zeros((len(config.instrument.frequency), 3, hp.nside2npix(config.nside),1))
        foregrounds[..., 0] = _load_input_foregrounds(config.fgds_path + f'/foregrounds_{"".join(config.foreground_models)}_{config.data_type}_ns{config.nside}' + (f"_lmin{config.lmin}" if config.lmin > 2 else ""))
    return foregrounds

def _get_data_simulations_(config: Configs, foregrounds, nsim = None):
    if nsim is not None:
        if not isinstance(nsim, int):
            raise ValueError("nsim must be an integer.")
            
    if config.generate_input_simulations:
        if not config.cls_cmb_path:
            # Define the path to the FITS file
            config.cls_cmb_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "data", "Cls_Planck2018_lensed_r0.fits")
        # Load the FITS file
        cls_cmb = hp.read_cl(config.cls_cmb_path)

        noise = _get_noise_simulation(config.instrument, config.nside, seed = None if not config.seed_noise else ((config.seed_noise + (nsim * 3 * len(config.instrument.frequency))) if nsim is not None else config.seed_noise), units=config.units,return_alms=(config.data_type=="alms"), lmin=config.lmin, ell_knee=config.ell_knee, alpha_knee=config.alpha_knee)
        if config.input_beams == "guassian":
            cmb = _get_cmb_simulation(cls_cmb, config.instrument, config.nside, seed = None if not config.seed_cmb else (config.seed_cmb + nsim if nsim is not None else config.seed_cmb), pixel_window = config.pixel_window_in, return_alms = (config.data_type=="alms"), lmin=config.lmin)
        else:
            bl_path = os.path.join(config.inputs_path, "beams", config.experiment)
            cmb = _get_cmb_simulation(cls_cmb, config.instrument, config.nside, seed = None if not config.seed_cmb else (config.seed_cmb + nsim if nsim is not None else config.seed_cmb), pixel_window = config.pixel_window_in, return_alms = (config.data_type=="alms"), lmin=config.lmin, bl_path=bl_path, symmetric_beam=config.input_beams=="file_lm")

        data = noise + cmb + foregrounds
        if config.save_input_simulations:
            _save_input_simulations(config, data, nsim=nsim, tag="total")
            _save_input_simulations(config, noise, nsim=nsim, tag="noise")
            _save_input_simulations(config, cmb, nsim=nsim, tag="cmb")
    elif config.load_input_simulations:
        noise = _load_input_simulations(config.noise_path, nsim=nsim)
        cmb = _load_input_simulations(config.cmb_path, nsim=nsim)
        if config.data_path:
            data = _load_input_simulations(config.data_path, nsim=nsim)
        else:
            data = noise + cmb + foregrounds
    return data, cmb, noise

def _save_input_simulations(config: Configs, maps, nsim = None, tag=""):
    path = os.path.join(config.inputs_path, config.experiment, tag)
    os.makedirs(path, exist_ok=True)
    filename = path + f"/{tag}_{config.data_type}_ns{config.nside}"
    if tag == "noise":
        if (config.ell_knee is not None) and (config.alpha_knee is not None):
            filename += f"_lk{config.ell_knee}_ak{config.alpha_knee}"
    if config.lmin > 2:
        filename += f"_lmin{config.lmin}"
    if nsim is not None:
        filename += f"_{str(nsim).zfill(5)}"
    np.save(filename, maps)

def _load_input_simulations(path, nsim = None):
    return np.load(path + f"_{str(nsim).zfill(5)}.npy") if nsim is not None else np.load(path + '.npy')
    
def _save_input_foregrounds(config: Configs, foregrounds):
    fg_path = os.path.join(config.inputs_path, config.experiment, "foregrounds", "".join(config.foreground_models))
    os.makedirs(fg_path, exist_ok=True)
    np.save(fg_path + f'/foregrounds_{"".join(config.foreground_models)}_{config.data_type}_ns{config.nside}' + (f"_lmin{config.lmin}" if config.lmin > 2 else ""), foregrounds[..., 0])
    if foregrounds.shape[-1] > 1:
        for idx, fmodel in enumerate(config.foreground_models):
            np.save(fg_path + f"/foregrounds_{fmodel}_{config.data_type}_ns{config.nside}" + (f"_lmin{config.lmin}" if config.lmin > 2 else ""), foregrounds[..., idx + 1])

def _load_input_foregrounds(fgd_path):
    return np.load(fgd_path + '.npy')

def _get_noise_simulation(instrument, nside, seed = None, units='uK_CMB',return_alms=False, lmin=2, ell_knee = None, alpha_knee = None):
    """
    Generate noise simulation for the given instrument.
    """
    if not hasattr(instrument, 'depth_I') and not hasattr(instrument, 'depth_P'):
        raise ValueError('Provided instrumental setting must have either depth_I or depth_P attributes.')
    elif not hasattr(instrument, 'depth_I') and hasattr(instrument, 'depth_P'):
        instrument.depth_I = instrument.depth_P / np.sqrt(2)
        raise Warning('No intensity map depth provided. Assuming it to be the polarization one divided by sqrt(2).')
    elif not hasattr(instrument, 'depth_P') and hasattr(instrument, 'depth_I'):
        instrument.depth_P = instrument.depth_I * np.sqrt(2)
        raise Warning('No polarization map depth provided. Assuming it to be the intensity one multiplied by sqrt(2).')
    
    depth_i = instrument.depth_I
    depth_p = instrument.depth_P
    depth_i *= u.arcmin * u.uK_CMB
    depth_i = depth_i.to(getattr(u, units) * u.arcmin, equivalencies=u.cmb_equivalencies(instrument.frequency * u.GHz))
    depth_p *= u.arcmin * u.uK_CMB
    depth_p = depth_p.to(getattr(u, units) * u.arcmin, equivalencies=u.cmb_equivalencies(instrument.frequency * u.GHz))

    acm_to_rad = (np.pi / (180 * 60)) 
    if lmin > 2:
        fell = _get_ell_filter(lmin,3*nside-1)

    noise = []
    for nf in range(len(instrument.frequency)):
        if seed is not None:
            np.random.seed(seed + (nf * 3))
        N_ell_T = (depth_i.value[nf] * acm_to_rad) ** 2 * np.ones(3 * nside)
        N_ell_P = (depth_p.value[nf] * acm_to_rad) ** 2 * np.ones(3 * nside)
        N_ell = np.array([N_ell_T, N_ell_P, N_ell_P, 0.*N_ell_P])
        if (alpha_knee is not None) and (ell_knee is not None):
            ell = np.arange(3 * nside)
            if np.isscalar(alpha_knee) and np.isscalar(ell_knee):
                N_ell *= (1 + (ell / ell_knee) ** alpha_knee)
            elif isinstance(alpha_knee, list) and isinstance(ell_knee, list):
                if len(alpha_knee) != len(ell_knee):
                    raise ValueError('alpha_knee and ell_knee must have the same length.')
                if (len(alpha_knee) != len(instrument.frequency)) or (len(ell_knee) != len(instrument.frequency)):
                    raise ValueError('alpha_knee and ell_knee must have the same length as the number of frequencies.')
                N_ell *= (1 + (ell / ell_knee[nf]) ** alpha_knee[nf])
            else:
                raise ValueError('alpha_knee and ell_knee must be either both scalars or both lists')
        alm_noise = hp.synalm(N_ell, lmax=3*nside-1, new=True, verbose=False)
        if lmin > 2:
            for f in range(3):
                alm_noise[f] = hp.almxfl(alm_noise[f], fell)
        if return_alms:
            noise.append(alm_noise)
        else:
            noise.append(hp.alm2map(alm_noise, nside, lmax=3*nside-1, pol=True))
    return np.array(noise)
#    if noise_type == "white":
#        noise = np.random.normal(size=(len(instrument.frequency), 3, hp.nside2npix(nside)))       
#        noise *= ((depth.value).T)[:, :, None]
#        noise /= hp.nside2resol(nside, True)
#        if return_alms:
#            return np.array([hp.map2alm(noise[i], lmax=3*nside-1, pol=True) for i in range(len(instrument.frequency))])
#        else:
#            return noise

def _get_cmb_simulation(cls_cmb, instrument, nside, seed = None, pixel_window = False, new = True, return_alms = False, lmin=2, bl_path=None, symmetric_beam=True):
    """
    Generate CMB simulation for the given instrument.
    """
    alm_cmb = _get_cmb_alms_realization(cls_cmb, 3 * nside - 1, seed = seed, new = new)
    if lmin > 2:
        fell = _get_ell_filter(lmin,3*nside-1)

    cmb = []
    for idx in range(len(instrument.frequency)):
        if bl_path is None:
            alm_cmb_i = _smooth_input_alms_(alm_cmb, fwhm=instrument.fwhm[idx], nside_out=nside if pixel_window else None)
        else:
            beamfile = bl_path + f"/beam_TEB_{instrument.frequency[idx]}GHz.fits"
            alm_cmb_i = _smooth_input_alms_(alm_cmb, beam_path=beamfile, symmetric_beam=symmetric_beam, nside_out=nside if pixel_window else None)

        if lmin > 2:
            for f in range(3):
                alm_cmb_i[f] = hp.almxfl(alm_cmb_i[f], fell)
        cmb.append(alm_cmb_i if return_alms else hp.alm2map(alm_cmb_i, nside, lmax = 3 * nside - 1, pol = True))
    return np.array(cmb)

def _get_cmb_alms_realization(cls_cmb, lmax, seed = None, new = True):
    """
    Generate a CMB realization
    """
    if seed is not None:
        np.random.seed(seed)
    cmb_alms = hp.synalm(cls_cmb, lmax=lmax, new=new, verbose=False)
    return cmb_alms

def _get_foregrounds(foreground_models, instrument, nside, return_components=False, pixel_window=False, units='uK_CMB', return_alms=False, bandpass_integrate=False, lmin=2, bl_path=None, symmetric_beam=True):
    if nside <= 512:
        nside_ = 512
    else:
        nside_ = nside
            
    if not return_components or len(foreground_models) == 1:
        sky = pysm3.Sky(nside=nside_, preset_strings=foreground_models, output_unit=getattr(u, units))
        foregrounds = _get_foreground_component(instrument, sky, nside, pixel_window=pixel_window, bandpass_integrate=bandpass_integrate, return_alms=return_alms, lmin=lmin, bl_path=bl_path, symmetric_beam=symmetric_beam)
        foregrounds = foregrounds[..., None]
    else:
        if return_alms:
            foregrounds = np.zeros((len(instrument.frequency), 3, hp.Alm.getsize(3*nside-1), len(foreground_models)+1), dtype=complex)
        else:
            foregrounds = np.zeros((len(instrument.frequency), 3, hp.nside2npix(nside), len(foreground_models)+1))
        for idx, fmodel in enumerate(foreground_models):
            sky = pysm3.Sky(nside=nside_, preset_strings=[fmodel], output_unit=getattr(u, units))
            foregrounds[:, :, :, idx + 1] = _get_foreground_component(instrument, sky, nside, pixel_window=pixel_window, bandpass_integrate=bandpass_integrate, return_alms=return_alms, lmin=lmin, bl_path=bl_path, symmetric_beam=symmetric_beam)
        foregrounds[:, :, :, 0] = np.sum(foregrounds[:, :, :, 1:], axis=-1)
    return foregrounds

def _get_foreground_component(instrument, sky, nside_out, pixel_window=False, bandpass_integrate=False, return_alms=False, lmin=2, bl_path=None, symmetric_beam=True):
    """ 
    """
    fg_component = []
    if lmin > 2:
        fell = _get_ell_filter(lmin,3*nside_out-1)
    
    for idx_freq, freq in enumerate(instrument.frequency):
        if bandpass_integrate:
            freq_min = freq * (1 - ( instrument.bandwidth[idx_freq] / 2 ))
            freq_max = freq * (1 + ( instrument.bandwidth[idx_freq] / 2 ))
            steps = int(freq_max - freq_min + 1)
            frequencies = np.linspace(freq_min, freq_max, steps) * u.GHz
            bandpass_weights = np.ones(len(frequencies)) # The tophat is defined in intensity units (Jy/sr)
            emission = sky.get_emission(frequencies, bandpass_weights)
        else:
            emission = sky.get_emission(freq * u.GHz)

        alm_emission = hp.map2alm(emission.value, lmax=3*nside_out-1, pol=True)
        
        if bl_path is None:
            alm_emission = _smooth_input_alms_(alm_emission, fwhm=instrument.fwhm[idx_freq], nside_out=nside_out if pixel_window else None)
        else:
            beamfile = bl_path + f"/beam_TEB_{freq}GHz.fits"
            alm_emission = _smooth_input_alms_(alm_emission, beam_path=beamfile, symmetric_beam=symmetric_beam, nside_out=nside_out if pixel_window else None)

        if lmin > 2:
            for f in range(3):
                alm_emission[f] = hp.almxfl(alm_emission[f], fell)
        fg_component.append(alm_emission if return_alms else hp.alm2map(alm_emission, nside_out, lmax=3*nside_out-1, pol=True))
        
    return np.array(fg_component)

def _smooth_input_alms_(alms, fwhm=None, nside_out=None, beam_path=None, symmetric_beam=True):
    alms_smoothed = np.zeros_like(alms)

    if fwhm is not None:
        bl_i = hp.gauss_beam(np.radians(fwhm/60.), lmax = hp.Alm.getlmax(alms.shape[1]), pol = True)
    elif beam_path is not None:
        bl_i = _get_beam_from_file(beam_path,hp.Alm.getlmax(alms.shape[1]),symmetric_beam=symmetric_beam)
    bl_i = bl_i[:,:3]

    if nside_out:
        pw = hp.pixwin(nside_out, pol=True, lmax=hp.Alm.getlmax(alms.shape[1]))
        pw = np.array([pw[0], pw[1], pw[1]])
        if symmetric_beam:
            bl_i = bl_i * pw.T
        else:
            for i in range(3):
                bl_i[:,i] = hp.almxfl(bl_i[:,i], pw[i])

    for i in range(3):
        if symmetric_beam:
            alms_smoothed[i] = hp.almxfl(alms[i], bl_i[:,i])
        else:
            alms_smoothed[i] = alms[i] * bl_i[:,i]

    return alms_smoothed

