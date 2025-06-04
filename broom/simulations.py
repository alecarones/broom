import numpy as np 
import healpy as hp
import pysm3.units as u
import os
import pysm3
from astropy.io import fits
from .configurations import Configs
from .routines import _get_ell_filter, _get_beam_from_file
from types import SimpleNamespace

def _get_full_simulations(config: Configs, nsim=None):
    foregrounds = _get_data_foregrounds_(config)
    data = _get_data_simulations_(config, foregrounds, nsim=nsim)
    return data

def _get_data_foregrounds_(config: Configs):
    if config.generate_input_foregrounds:
        if config.verbose:
            print(f"Generating foreground maps of {''.join(config.foreground_models)} model" + " with bandpass integration" if config.bandpass_integrate else "")
        foregrounds = _get_foregrounds(config.foreground_models, config.instrument, config.nside, config.lmax, return_components=config.return_fgd_components, pixel_window=config.pixel_window_in, units=config.units, return_alms=(config.data_type=="alms"), bandpass_integrate=config.bandpass_integrate, lmin=config.lmin, coordinates=config.coordinates)
        if config.save_inputs:
            if config.verbose:
                print(f"Saving foreground maps in {config.fgds_path} directory")
            _save_input_foregrounds(config.fgds_path, foregrounds, config.foreground_models)
    else:
        foregrounds = SimpleNamespace()
        if config.return_fgd_components:
            prefix_to_attr = {"d": "dust", "s": "synch", "a": "ame", "co": "co", "f": "freefree", "cib": "cib", "tsz": "tsz", "ksz": "ksz", "rg": "radio_galaxies"}
            for fmodel in config.foreground_models:
                attr = prefix_to_attr.get(fmodel[:2]) or prefix_to_attr.get(fmodel[:1]) or prefix_to_attr.get(fmodel[:3])
                setattr(foregrounds, attr, _load_input_foregrounds(config.fgds_path, fmodel))
        foregrounds.total = _load_input_foregrounds(config.fgds_path, "".join(config.foreground_models))
    return foregrounds

def _get_data_simulations_(config: Configs, foregrounds = None, nsim = None):
    if nsim is not None:
        if not isinstance(nsim, (int,str)):
            raise ValueError("nsim must be an integer or a string.")
        if isinstance(nsim, int):
            nsim = str(nsim).zfill(5)

    if foregrounds is not None:
        if not hasattr(foregrounds, 'total'):
            raise ValueError('foregrounds must have the attribute total.')

    data =  SimpleNamespace()

    if config.generate_input_cmb:
        data.cmb = _get_cmb_simulation(config, nsim=nsim)
    else:
        if config.cmb_path is not None:
            if config.verbose:
                print(f"Loading CMB from {config.cmb_path}.npy file" if nsim is None else f"Loading CMB from {config.cmb_path}_{nsim}.npy")
            data.cmb = _load_inputs(config.cmb_path, nsim=nsim)

    if config.generate_input_noise:
        data.noise = _get_noise_simulation(config, nsim=nsim)
    else:
        if config.noise_path is not None:
            if config.verbose:
                print(f"Loading noise from {config.noise_path}.npy file" if nsim is None else f"Loading noise from {config.noise_path}_{nsim}.npy")
            data.noise = _load_inputs(config.noise_path, nsim=nsim)

    if config.generate_input_data:
        if (hasattr(data, 'cmb')) and (hasattr(data, 'noise')) and (foregrounds is not None):
            data.total = data.noise + data.cmb + foregrounds.total
            if config.save_inputs:
                _save_inputs(config.data_path, data.total, nsim=nsim)
        else:
            raise ValueError("To generate input data, you must provide foregrounds and either CMB and noise paths or generate them.")
    else:
        data.total = _load_inputs(config.data_path, nsim=nsim)

    if foregrounds is not None:
        data.fgds = foregrounds.total

    return data

def _save_inputs(filename, maps, nsim = None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if nsim is not None:
        filename += f"_{nsim}"
    np.save(filename, maps)

def _load_inputs(path, nsim = None):
    return np.load(path + f"_{nsim}.npy") if nsim is not None else np.load(path + '.npy')
    
def _save_input_foregrounds(fgds_path, foregrounds, foreground_models):
    os.makedirs(os.path.dirname(fgds_path), exist_ok=True)
    np.save(fgds_path + f'_{"".join(foreground_models)}', foregrounds.total)
    if len(vars(foregrounds)) > 1:
        prefix_to_attr = {"d": "dust", "s": "synch", "a": "ame", "co": "co", "f": "freefree", "cib": "cib", "tsz": "tsz", "ksz": "ksz", "rg": "radio_galaxies"}
        if len(vars(foregrounds)) != len(foreground_models) + 1:
            raise ValueError('The number of foreground models must match the number of foreground components.')
        for fmodel in foreground_models:
            attr = prefix_to_attr.get(fmodel[:2]) or prefix_to_attr.get(fmodel[:1]) or prefix_to_attr.get(fmodel[:3])
            np.save(fgds_path + f"_{fmodel}", foregrounds.__dict__[attr])

def _load_input_foregrounds(fgd_path, fgd_model):
    return np.load(f'{fgd_path}_{fgd_model}.npy')

def _get_noise_simulation(config: Configs, nsim=None):
    """
    Generate noise simulation for the given instrument.
    """
    seed = None if not config.seed_noise else ((config.seed_noise + (int(nsim) * 3 * len(config.instrument.frequency))) if nsim is not None else config.seed_noise)

    if not hasattr(config.instrument, 'path_depth_maps'):
        if not hasattr(config.instrument, 'depth_I') and not hasattr(config.instrument, 'depth_P'):
            raise ValueError('Provided instrumental setting must have either depth_I or depth_P attributes.')
        elif not hasattr(config.instrument, 'depth_I') and hasattr(config.instrument, 'depth_P'):
            config.instrument.depth_I = config.instrument.depth_P / np.sqrt(2)
            raise Warning('No intensity map depth provided. Assuming it to be the polarization one divided by sqrt(2).')
        elif not hasattr(config.instrument, 'depth_P') and hasattr(config.instrument, 'depth_I'):
            config.instrument.depth_P = config.instrument.depth_I * np.sqrt(2)
            raise Warning('No polarization map depth provided. Assuming it to be the intensity one multiplied by sqrt(2).')
        depth_i = config.instrument.depth_I
        depth_p = config.instrument.depth_P

        if hasattr(config.instrument, 'path_hits_maps'):
            if config.instrument.path_hits_maps[-5:] == ".fits":
                hits_map = hp.read_map(config.instrument.path_hits_maps, field=0, dtype=np.float64)
                if hp.get_nside(hits_map) != config.nside:
                    hits_map = hp.ud_grade(hits_map, nside_out=config.nside, power=-2)
                hits_map /= np.amax(hits_map)
    else:
        depth_i = [1.] * len(config.instrument.frequency)
        depth_p = [1.] * len(config.instrument.frequency)
        
    depth_i *= u.arcmin * u.uK_CMB
    depth_i = depth_i.to(getattr(u, config.units) * u.arcmin, equivalencies=u.cmb_equivalencies(config.instrument.frequency * u.GHz))
    depth_p *= u.arcmin * u.uK_CMB
    depth_p = depth_p.to(getattr(u, config.units) * u.arcmin, equivalencies=u.cmb_equivalencies(config.instrument.frequency * u.GHz))

    acm_to_rad = (np.pi / (180 * 60)) 
    if config.lmin > 2:
        fell = _get_ell_filter(config.lmin,config.lmax)

    noise = []
    for nf in range(len(config.instrument.frequency)):
        if hasattr(config.instrument, 'path_depth_maps'):
            depth_map_fn = os.path.join(config.instrument.path_depth_maps, f"depth_map_{config.instrument.channels_tags[nf]}.fits")
            try:
                depth_maps_in = hp.read_map(depth_map_fn, field=(0,1), dtype=np.float64)
            except IndexError:
                print("Warning: Unable to read depth maps from the provided path for I and P, provided depth map is assumed to refer to polarization.")
                depth_maps_in = hp.read_map(depth_map_fn, field=0, dtype=np.float64)
                depth_maps_in = np.array([depth_maps_in / np.sqrt(2), depth_maps_in])
            if hp.get_nside(depth_maps_in[0]) != config.nside:
                depth_maps = []
                for depth_map in depth_maps_in:
                    depth_maps.append(np.sqrt(hp.ud_grade(depth_map**2, nside_out=config.nside, power=2)))
                depth_maps = np.array(depth_maps)
            else:
                depth_maps = np.copy(depth_maps_in)
            del depth_maps_in
        elif hasattr(config.instrument, 'path_hits_maps'):
            if config.instrument.path_hits_maps[-5:] != ".fits":
                hits_file = os.path.join(config.instrument.path_hits_maps, f"hits_map_{config.instrument.channels_tags[nf]}.fits")
                hits_map = hp.read_map(hits_file, field=0, dtype=np.float64)
                if hp.get_nside(hits_map) != config.nside:
                    hits_map = hp.ud_grade(hits_map, nside_out=config.nside, power=-2)
                hits_map /= np.amax(hits_map)

        if seed is not None:
            np.random.seed(seed + (nf * 3))
        N_ell_T = (depth_i.value[nf] * acm_to_rad) ** 2 * np.ones(config.lmax + 1)
        N_ell_P = (depth_p.value[nf] * acm_to_rad) ** 2 * np.ones(config.lmax + 1)
        N_ell = np.array([N_ell_T, N_ell_P, N_ell_P, 0.*N_ell_P])
        if hasattr(config.instrument, 'ell_knee') and hasattr(config.instrument, 'alpha_knee'):
            ell = np.arange(config.lmax + 1)
            if isinstance(config.instrument.alpha_knee, list) and isinstance(config.instrument.ell_knee, list):
                if len(config.instrument.alpha_knee) != len(config.instrument.ell_knee):
                    raise ValueError('alpha_knee and ell_knee must have the same length.')
                if (len(config.instrument.alpha_knee) != len(config.instrument.frequency)) or (len(config.instrument.ell_knee) != len(config.instrument.frequency)):
                    raise ValueError('alpha_knee and ell_knee must have the same length as the number of frequencies.')
                N_ell *= (1 + (ell / config.instrument.ell_knee[nf]) ** config.instrument.alpha_knee[nf])
            else:
                raise ValueError('alpha_knee and ell_knee must be both lists')
        alm_noise = hp.synalm(N_ell, lmax=config.lmax, new=True)
        if config.lmin > 2:
            for f in range(3):
                alm_noise[f] = hp.almxfl(alm_noise[f], fell)
        if config.data_type=="alms":
            if hasattr(config.instrument, 'path_depth_maps'):
                noise_map = hp.alm2map(alm_noise, config.nside, lmax=config.lmax, pol=True)  * np.array([depth_maps[0], depth_maps[1], depth_maps[1]])
                noise.append(hp.map2alm(noise_map, lmax=config.lmax, pol=True))
            elif hasattr(config.instrument, 'path_hits_maps'):
                noise_map = hp.alm2map(alm_noise, config.nside, lmax=config.lmax, pol=True) / np.sqrt(hits_map)
                noise_map[np.where(np.abs(noise_map_split)==float("inf"))] = 0.
                noise.append(hp.map2alm(noise_map, lmax=config.lmax, pol=True))
            else:
                noise.append(alm_noise)
        else:
            if hasattr(config.instrument, 'path_depth_maps'):
                noise.append(hp.alm2map(alm_noise, config.nside, lmax=config.lmax, pol=True) * np.array([depth_maps[0], depth_maps[1], depth_maps[1]]))
            elif hasattr(config.instrument, 'path_hits_maps'):
                noise_map = hp.alm2map(alm_noise, config.nside, lmax=config.lmax, pol=True) / np.sqrt(hits_map)
                noise_map[np.where(np.abs(noise_map)==float("inf"))] = 0.
                noise.append(noise_map)
            else:
                noise.append(hp.alm2map(alm_noise, config.nside, lmax=config.lmax, pol=True))

    if config.save_inputs:
        _save_inputs(config.noise_path, np.array(noise), nsim=nsim)

    return np.array(noise)
#    if noise_type == "white":
#        noise = np.random.normal(size=(len(instrument.frequency), 3, hp.nside2npix(nside)))       
#        noise *= ((depth.value).T)[:, :, None]
#        noise /= hp.nside2resol(nside, True)
#        if return_alms:
#            return np.array([hp.map2alm(noise[i], lmax=3*nside-1, pol=True) for i in range(len(instrument.frequency))])
#        else:
#            return noise

def _get_cmb_simulation(config: Configs, nsim=None, new = True):
    """
    Generate CMB simulation for the given instrument.
    """
    if not config.cls_cmb_path:
        # Define the path to the FITS file
        config.cls_cmb_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "utils", "Cls_Planck2018_lensed_r0.fits")
    # Load the FITS file
    cls_cmb = hp.read_cl(config.cls_cmb_path)

    seed = None if not config.seed_cmb else (config.seed_cmb + int(nsim) if nsim is not None else config.seed_cmb)
        
    alm_cmb = _get_cmb_alms_realization(cls_cmb, config.lmax, seed = seed, new = new)
    if config.lmin > 2:
        fell = _get_ell_filter(config.lmin,config.lmax)

    cmb = []
    for idx in range(len(config.instrument.frequency)):
        if config.instrument.beams == "gaussian":
            alm_cmb_i = _smooth_input_alms_(alm_cmb, fwhm=config.instrument.fwhm[idx], nside_out=config.nside if config.pixel_window_in else None)
        else:
            beamfile = config.instrument.path_beams + f"/beam_TEB_{config.instrument.channels_tags[idx]}.fits"
            alm_cmb_i = _smooth_input_alms_(alm_cmb, beam_path=beamfile, symmetric_beam=config.instrument.input_beams=="file_l", nside_out=config.nside if config.pixel_window_in else None)

        if config.lmin > 2:
            for f in range(3):
                alm_cmb_i[f] = hp.almxfl(alm_cmb_i[f], fell)
        cmb.append(alm_cmb_i if config.data_type=="alms" else hp.alm2map(alm_cmb_i, config.nside, lmax = config.lmax, pol = True))
    cmb = np.array(cmb)

    if config.save_inputs:
        _save_inputs(config.cmb_path, cmb, nsim=nsim)
    return cmb

def _get_cmb_alms_realization(cls_cmb, lmax, seed = None, new = True):
    """
    Generate a CMB realization
    """
    if seed is not None:
        np.random.seed(seed)
    cmb_alms = hp.synalm(cls_cmb, lmax=lmax, new=new)
    return cmb_alms

def _get_foregrounds(foreground_models, instrument, nside, lmax, return_components=False, pixel_window=False, units='uK_CMB', return_alms=False, bandpass_integrate=False, lmin=2, coordinates="G"):
    if nside <= 512:
        nside_ = 512
    else:
        nside_ = nside
    
    foregrounds = SimpleNamespace()

    if not return_components or len(foreground_models) == 1:
        sky = pysm3.Sky(nside=nside_, preset_strings=foreground_models, output_unit=getattr(u, units))
        foregrounds.total = _get_foreground_component(instrument, sky, nside, lmax, pixel_window=pixel_window, bandpass_integrate=bandpass_integrate, return_alms=return_alms, lmin=lmin, coordinates=coordinates)
    else:
        prefix_to_attr = {"d": "dust", "s": "synch", "a": "ame", "co": "co", "f": "freefree", "cib": "cib", "tsz": "tsz", "ksz": "ksz", "rg": "radio_galaxies"}
        for fmodel in foreground_models:
            sky = pysm3.Sky(nside=nside_, preset_strings=[fmodel], output_unit=getattr(u, units))
            attr = prefix_to_attr.get(fmodel[:2]) or prefix_to_attr.get(fmodel[:1]) or prefix_to_attr.get(fmodel[:3])
            setattr(foregrounds, attr, _get_foreground_component(instrument, sky, nside, lmax, pixel_window=pixel_window, bandpass_integrate=bandpass_integrate, return_alms=return_alms, lmin=lmin, coordinates=coordinates))
        foregrounds.total = sum(vars(foregrounds).values())
    return foregrounds

def _get_foreground_component(instrument, sky, nside_out, lmax, pixel_window=False, bandpass_integrate=False, return_alms=False, lmin=2, coordinates="G"):
    """ 
    """
    fg_component = []
    if lmin > 2:
        fell = _get_ell_filter(lmin,lmax)

    if coordinates != "G":
        rot=hp.Rotator(coord=f"G{coordinates}")
    
    for idx_freq, freq in enumerate(instrument.frequency):
        if bandpass_integrate:
            if hasattr(instrument, 'path_bandpasses'):
                frequencies, bandpass_weights = np.load(os.path.join(instrument.path_bandpasses, f"bandpass_{instrument.channels_tags[idx_freq]}.npy"))
            else:
                freq_min = freq * (1 - ( instrument.bandwidth[idx_freq] / 2 ))
                freq_max = freq * (1 + ( instrument.bandwidth[idx_freq] / 2 ))
                steps = int(freq_max - freq_min + 1)
                frequencies = np.linspace(freq_min, freq_max, steps) * u.GHz
                bandpass_weights = np.ones(len(frequencies)) # The tophat is defined in intensity units (Jy/sr)
            emission = sky.get_emission(frequencies, bandpass_weights)
        else:
            emission = sky.get_emission(freq * u.GHz)

#        if coordinates != "G":
#            emission = pysm3.apply_smoothing_and_coord_transform(emission, rot=hp.Rotator(coord=f"G{coordinates}"), lmax=3*nside_out-1)

        alm_emission = hp.map2alm(emission.value, lmax=lmax, pol=True)
        if coordinates != "G":
            rot.rotate_alm(alm_emission, inplace=True)
        
        if instrument.beams == "gaussian":
            alm_emission = _smooth_input_alms_(alm_emission, fwhm=instrument.fwhm[idx_freq], nside_out=nside_out if pixel_window else None)
        else:
            beamfile = instrument.path_beams + f"/beam_TEB_{instrument.channels_tags[idx_freq]}.fits"
            alm_emission = _smooth_input_alms_(alm_emission, beam_path=beamfile, symmetric_beam=instrument.input_beams=="file_l", nside_out=nside_out if pixel_window else None)

        if lmin > 2:
            for f in range(3):
                alm_emission[f] = hp.almxfl(alm_emission[f], fell)
        fg_component.append(alm_emission if return_alms else hp.alm2map(alm_emission, nside_out, lmax=lmax, pol=True))
        
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

