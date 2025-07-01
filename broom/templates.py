import numpy as np
import healpy as hp
from .configurations import Configs
from .routines import obj_to_array, array_to_obj
from .saving import _save_residuals_template
from .needlets import _get_nside_lmax_from_b_ell, _get_needlet_windows_, _needlet_filtering
from .ilcs import _ilc_post_processing, _get_good_channels_nl
from types import SimpleNamespace
import os
from typing import Union, Dict, Any, List
import sys

def get_residuals_template(
    config: Configs,
    input_alms: SimpleNamespace,
    compsep_run: dict,
    **kwargs
) -> Union[None, SimpleNamespace]:
    """
    Estimates the foreground residuals template from input foregrounds multifrequency templates using 
    component separation weights as defined in compsep_run dictionary.

    Parameters
    ----------
        config : Configs
            Configuration object containing the global settings. It should include the following attributes:
            - nside: The HEALPix resolution parameter.
            - lmax: The maximum multipole for the analysis.
            - fwhm_out: The full width at half maximum for the foreground templates and at which weights
            were computed.
            - pixel_window_out: Boolean indicating whether to apply pixel windowing to the output maps.
            - save_compsep_products: Boolean indicating whether to save the residuals templates.
            - return_compsep_products: Boolean indicating whether to return the residuals templates.
            - mask_type: Type of mask to apply, e.g., "observed_patch".
            - leakage_correction: Method for leakage correction, e.g., "purify", or "recycling". Used 
            only for PILC residuals estimation.
            - field_out: The field to be outputted, e.g., "E", "B", "EB", "QU", "QU_E", or "QU_B".
        
        input_alms : np.ndarray
            Input foregrounds multifrequency templates in alm format. The shape should be (n_freqs, n_fields, n_alms, n_comps) where:
            - n_freqs: Number of frequency channels.
            - n_fields: Number of fields, not None only if multiple fields are being processed simultaneously.
            - n_alms: Number of spherical harmonic coefficients.
            - n_comps: Number of components referring to foreground templates (e.g. total, noise residuals, etc.).
        
        compsep_run : dict
            Dictionary containing the residuals estimates run parameters. It should include:
            - compsep_path: Path to the component separation weights.
            - field_in_cs: The field associated to input templates and component separation weights, e.g., "T", "E", "B", "QU_E", "QU_B".
            - nsim: Simulation number if applicable, otherwise None.
            - adapt_nside: Boolean indicating whether to adapt nside based on the needlet bands.
            - mask: Optional mask to apply to the output maps.
        
        kwargs : dict
            Additional keywords for healpy map2alm function.

    Returns
    -------
        SimpleNamespace or None
            Estimate of foreground residuals for component separation of scalar fields or for PILC.
            It may have the following attributes:
            - fgres_templates: Total foreground residuals.
            - fgres_templates_noise: Noise contamination in the foreground residuals template.
            - fgres_templates_ideal: Pure foreground component in the template.
            Each attribute has shape (n_fields, n_pixels) with n_fields depending on config.field_out.
    """

    templates = _get_fres(config, obj_to_array(input_alms), compsep_run, **kwargs)
    
    if "pilc" not in compsep_run["compsep_path"]:
        templates = _ilc_post_processing(config, templates, compsep_run, **kwargs)

    templates = array_to_obj(templates, input_alms)

    if config.save_compsep_products:
        _save_residuals_template(config, templates, compsep_run, nsim=compsep_run["nsim"])

    if config.return_compsep_products:
        return templates
    return None

def _get_fres(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: dict,
    **kwargs
) -> np.ndarray:
    """
    Estimates the foreground residuals template from input foregrounds multifrequency templates using
    component separation weights as defined in compsep_run dictionary.

    Parameters
    ----------
        config : Configs
            Configuration object containing the global settings. See 'get_residuals_template' for details.
        input_alms : np.ndarray
            Input foregrounds multifrequency templates in alm format. The shape should be (n_freqs, n_fields, n_alms, n_comps).
            n_fields is not None only if multiple fields are being processed simultaneously.
        compsep_run : dict
            Dictionary containing the residuals estimates run parameters. See 'get_residuals_template' for details.
        kwargs : dict
            Additional keywords for healpy map2alm function.
    
    Returns
    -------
        np.ndarray
            Estimate of foreground residuals for component separation of scalar fields.
            The shape will be (n_fields, n_pixels, n_comps) with n_fields depending on config.field_out.
    """

    if "pilc" in compsep_run["compsep_path"]:
        if input_alms.ndim == 4:
            if input_alms.shape[1] == 3:
                raise ValueError("PILC does not support TEB alms, only EB, E or B alms.")
            elif input_alms.shape[1] == 2:
                compsep_run["field"] = "QU"
        elif input_alms.ndim == 3:
            if compsep_run["field_in_cs"] in ["E", "B"]:
                compsep_run["field"] = f'QU_{compsep_run["field_in_cs"]}'
            elif compsep_run["field_in_cs"] in ["QU_E", "QU_B"]:
                compsep_run["field"] = compsep_run["field_in_cs"]
            else:
                raise ValueError(f"Field {compsep_run['field_in_cs']} not supported for PILC residuals estimate.")
        output_maps = _get_fres_P(config, input_alms, compsep_run, **kwargs)
    else:
        if input_alms.ndim == 4:
            fields_ilc = ["T", "E", "B"] if input_alms.shape[1] == 3 else ["E", "B"]
        elif input_alms.ndim == 3:
            if compsep_run["field_in_cs"] in ["T", "E", "B"]:
                fields_ilc = [compsep_run["field_in_cs"]]
            elif compsep_run["field_in_cs"] in ["QU_E", "QU_B"]:
                fields_ilc = [compsep_run["field_in_cs"][-1]]

        if input_alms.ndim == 4:
            output_maps = np.zeros((input_alms.shape[1], 12 * config.nside**2, input_alms.shape[3]))
            for i in range(input_alms.shape[1]):
                compsep_run["field"] = fields_ilc[i]
                output_maps[i] = _get_fres_scalar(config, input_alms[:, i, :, :], compsep_run, **kwargs)
        elif input_alms.ndim == 3:
            compsep_run["field"] = fields_ilc[0]
            output_maps = _get_fres_scalar(config, input_alms, compsep_run, **kwargs)
    
    del compsep_run["field"]

    return output_maps

def _get_fres_scalar(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: dict,
    **kwargs: Any
) -> np.ndarray:
    """
    Estimates the foreground residuals template from input foregrounds multifrequency templates using
    component separation weights as defined in compsep_run dictionary for a single scalar field.

    Parameters
    ----------
        config : Configs
            Configuration object containing the global settings. See 'get_residuals_template' for details.
        input_alms : np.ndarray
            Input foregrounds multifrequency templates in alm format for a single scalar field.
            The shape should be (n_freqs, n_alms, n_comps).
        compsep_run : dict
            Dictionary containing the residuals estimates run parameters. See 'get_residuals_template' for details.
        kwargs : dict
            Additional keywords for healpy map2alm function.
    
    Returns
    -------
        np.ndarray
        Estimate of foreground residuals for component separation of scalar fields in pixel or needlet domain.
        The shape will be (n_pixels, n_comps).
    """

    domain = compsep_run["compsep_path"].split('ilc_')[1].split('_bias')[0]
    if domain == "pixel":
        return _get_fres_pixel(config, input_alms, compsep_run, **kwargs)
    elif domain == "needlet":
        return _get_fres_needlet(config, input_alms, compsep_run, **kwargs)
    else:
        raise ValueError(f"Domain {domain} not supported for foreground residuals estimation.")

def _get_fres_P(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: dict,
    **kwargs: Any
) -> np.ndarray:
    """
    Estimates the foreground residuals template from input foregrounds multifrequency templates using
    component separation PILC component separation weights.

    Parameters
    ----------
        config : Configs
            Configuration object containing the global settings. See 'get_residuals_template' for details.  
        input_alms : np.ndarray
            Input foregrounds multifrequency templates in alm format for polarization field.
            The shape should be (n_freqs, (n_fields), n_alms, n_comps), with n_fields not None if both E and B fields are present.
        compsep_run : dict
            Dictionary containing the residuals estimates run parameters. See 'get_residuals_template' for details.
        kwargs : dict
            Additional keywords for healpy map2alm function.

    Returns
    -------
        np.ndarray
            Estimate of foreground residuals from PILC. Shape is (n_fields, n_pixels, n_comps) where n_fields depends on config.field_out.
    """

    domain = compsep_run["compsep_path"].split('pilc_')[1].split('_bias')[0]
    if domain == "pixel":
        return _get_fres_P_pixel(config, input_alms, compsep_run, **kwargs)
    elif domain == "needlet":
        return _get_fres_P_needlet(config, input_alms, compsep_run, **kwargs)
    else:
        raise ValueError(f"Domain {domain} not supported for PILC foreground residuals estimation.")

def _get_fres_needlet(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: dict,
    **kwargs: Any
) -> np.ndarray:
    """
    Estimates the foreground residuals template from input foregrounds multifrequency templates using
    component separation weights in needlet domain for a single scalar field.

    Parameters
    ----------
        config : Configs
            Configuration object containing the global settings. See 'get_residuals_template' for details.  
        input_alms : np.ndarray
            Input foregrounds multifrequency templates in alm format.
            The shape should be (n_freqs, n_alms, n_comps).
        compsep_run : dict
            Dictionary containing the residuals estimates run parameters. See 'get_residuals_template' for details.
        kwargs : dict
            Additional keywords for healpy map2alm function.
        
    Returns
    -------
        np.ndarray
        Estimate of foreground residuals for component separation in needlet domain.
        Shape is (n_pixels, n_comps).
    """
    nl_filename = os.path.join(config.path_outputs, compsep_run["compsep_path"], "needlet_bands.npy")
    if os.path.exists(nl_filename):
        b_ell = np.load(nl_filename)
    else:
        raise ValueError(f"Needlet bands need to be saved as a npy file in {os.path.join(config.path_outputs, compsep_run['compsep_path'])}")

    output_alms = np.zeros((input_alms.shape[1], input_alms.shape[-1]), dtype=complex)
    for j in range(b_ell.shape[0]):
        output_alms += _get_fres_needlet_j(config, input_alms, compsep_run, b_ell[j], j, **kwargs)
    
    output_maps = np.array([
        hp.alm2map(
            np.ascontiguousarray(output_alms[:, c]),
            config.nside,
            lmax=config.lmax,
            pol=False,
            pixwin=config.pixel_window_out
        ) for c in range(input_alms.shape[-1])
    ]).T
    
    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.,:] = 0.

    return output_maps
        
def _get_fres_needlet_j(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: dict,
    b_ell: np.ndarray,
    nl_scale: int,
    **kwargs: Any
) -> np.ndarray:
    """
    Estimates the foreground residuals template from input foregrounds multifrequency templates using
    component separation weights for a single needlet band.

    Parameters
    ----------
        config : Configs
            Configuration object containing the global settings. See 'get_residuals_template' for details.  
        input_alms : np.ndarray
            Input foregrounds multifrequency templates in alm format.
            The shape should be (n_freqs, n_alms, n_comps).
        compsep_run : dict
            Dictionary containing the residuals estimates run parameters. See 'get_residuals_template' for details.
        b_ell : np.ndarray
            Needlet band to be used for the residuals estimation.
        nl_scale : int
            Scale of the needlet band, used to determine the weights filename.
        kwargs : dict
            Additional keywords for healpy map2alm function.
    
    Returns
    -------
        np.ndarray
            Estimate of foreground residuals for component separation in needlet domain for a single band.
            Shape is (n_pixels, n_comps).
    """

    if "mask" in compsep_run or not compsep_run["adapt_nside"]:
        nside_, lmax_ = config.nside, config.lmax
    else:
        nside_, lmax_ = _get_nside_lmax_from_b_ell(b_ell,config.nside,config.lmax)
        
    # Load weights
    weights_filename = os.path.join(config.path_outputs,
        compsep_run["compsep_path"],
        f"weights/weights_{compsep_run['field']}_{config.fwhm_out}acm_ns{nside_}_lmax{config.lmax}_nl{nl_scale}"
    )
    if compsep_run["nsim"] is not None:
        weights_filename += f"_{compsep_run['nsim']}"
    weights_filename += ".npy"

    w_ = np.load(weights_filename)

    # Select good channels and apply needlet filtering
    good_channels_nl = _get_good_channels_nl(config, b_ell)

    input_maps_nl = np.zeros((good_channels_nl.shape[0], 12 * nside_**2, input_alms.shape[-1]))
    for n, channel in enumerate(good_channels_nl):
        input_alms_j = _needlet_filtering(input_alms[channel], b_ell, lmax_)
        input_maps_nl[n] = np.array([
            hp.alm2map(np.ascontiguousarray(input_alms_j[:, c]), nside_, lmax=lmax_, pol=False)
            for c in range(input_alms.shape[-1])
        ]).T

    # Apply ILC weights
    if w_.ndim==1:
        output_maps_nl = np.einsum('i,ijk->jk', w_, input_maps_nl)
    elif w_.ndim==2:
        output_maps_nl = np.einsum('ij,ijk->jk', w_, input_maps_nl)

    del input_maps_nl

    # Convert to alms
    output_alms_nl = np.array([
        hp.map2alm(output_maps_nl[:, c], lmax=lmax_, pol=False, **kwargs)
        for c in range(output_maps_nl.shape[-1])
    ]).T

    nl_bands = compsep_run["compsep_path"].rstrip('/').split('/')[-1]

    if "nlsquared" in nl_bands:
        output_alms_j = _needlet_filtering(output_alms_nl, np.ones(lmax_+1), config.lmax)
    else:
        output_alms_j = _needlet_filtering(output_alms_nl, b_ell[:lmax_+1], config.lmax)

    return output_alms_j

def _get_fres_pixel(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: dict,
    **kwargs: Any
) -> np.ndarray:
    """
    Estimates the foreground residuals template from input foregrounds multifrequency templates using
    component separation weights in pixel domain for a single scalar field.

    Parameters
    ----------
        config : Configs
            Configuration object containing the global settings. See 'get_residuals_template' for details.
        input_alms : np.ndarray
            Input foregrounds multifrequency templates in alm format.
            The shape should be (n_freqs, n_alms, n_comps).
        compsep_run : dict
            Dictionary containing the residuals estimates run parameters. See 'get_residuals_template' for details.
        kwargs : dict
            Additional keywords for healpy map2alm function.
    
    Returns
    -------
        np.ndarray
            Estimate of foreground residuals for component separation of a single scalar field in pixel domain.
            Shape is (n_pixels, n_comps).
    """
    good_channels = _get_good_channels_nl(config, np.ones(config.lmax + 1))

    # Convert alms to maps
    input_maps = np.zeros((good_channels.shape[0], 12 * config.nside**2, input_alms.shape[-1]))
    for n, channel in enumerate(good_channels):
        input_maps[n] = np.array([
            hp.alm2map(np.ascontiguousarray(input_alms[channel, :, c]), config.nside, lmax=config.lmax, pol=False)
            for c in range(input_alms.shape[-1])
        ]).T

    # Load weights
    weights_filename = os.path.join(config.path_outputs,
        compsep_run["compsep_path"],
        f"weights/weights_{compsep_run['field']}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
    )
    if compsep_run["nsim"] is not None:
        weights_filename += f"_{compsep_run['nsim']}"
    weights_filename += ".npy"
    w_ = np.load(weights_filename)
    
    # Apply weights
    if w_.ndim==1:
        output_maps = np.einsum('i,ijk->jk', w_, input_maps)
    elif w_.ndim==2:
        output_maps = np.einsum('ij,ijk->jk', w_, input_maps)

    # Optionally apply pixel window function
    if config.pixel_window_out:
        for c in range(output_maps.shape[1]):
            alm_out = hp.map2alm(output_maps[:,c],lmax=config.lmax, pol=False, **kwargs)
            output_maps[:,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=False, pixwin=True)

    # Apply mask if available
    if "mask" in compsep_run:
        output_maps[compsep_run["mask"] == 0.,:] = 0.

    return output_maps

def _get_fres_P_needlet(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: dict,
    **kwargs: Any
) -> np.ndarray:
    """
    Estimates the foreground residuals template from input foregrounds multifrequency templates using
    PILC component separation weights in needlet domain.

    Parameters
    ----------
        config : Configs
            Configuration object containing the global settings. See 'get_residuals_template' for details.
        input_alms : np.ndarray
            Input foregrounds multifrequency templates in alm format for polarization field.
            The shape should be (n_freqs, (n_fields), n_alms, n_comps), with n_fields not None if both E and B fields are present.
        compsep_run : dict
            Dictionary containing the residuals estimates run parameters. See 'get_residuals_template' for details.
        kwargs : dict
            Additional keywords for healpy map2alm function.
    
    Returns
    -------
        np.ndarray
            Estimate of foreground residuals from PILC in needlet domain.
            Shape is (n_fields, n_pixels, n_comps) where n_fields depends on config.field_out.
    """
    # Load needlet bands    
    nl_filename = os.path.join(config.path_outputs, compsep_run["compsep_path"], "needlet_bands.npy")
    if os.path.exists(nl_filename):
        b_ell = np.load(nl_filename)
    else:
        raise ValueError(f"Needlet bands need to be saved as a .npy file in {os.path.join(config.path_outputs, compsep_run['compsep_path'])}")

    # Initialize output map
    output_maps = np.zeros((2, hp.nside2npix(config.nside), input_alms.shape[-1]))

    # Loop over needlet bands and compute residuals
    for j in range(b_ell.shape[0]):
        output_maps += _get_fres_P_needlet_j(config, input_alms, compsep_run, b_ell[j], j, **kwargs)
    
    process_outputs = (
        ((config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out) or
        (config.field_out in ["E", "B", "EB"])
    )

    if process_outputs:
        for c in range(output_maps.shape[-1]):
#            if "mask" in compsep_run and config.mask_type == "observed_patch" and config.leakage_correction is not None:
#                if "_purify" in config.leakage_correction:
#                    alm_out = purify_master(
#                        output_maps[..., c], compsep_run["mask"], config.lmax,
#                        purify_E=("E" in config.leakage_correction)
#                    )
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
#                elif "_recycling" in config.leakage_correction:
#                    iterations = (
#                        int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
#                        if "_iterations" in config.leakage_correction else 0
#                    )
#                    alm_out = purify_recycling(
#                        output_maps[..., c], output_maps[..., 0],
#                        np.ceil(compsep_run["mask"]), config.lmax,
#                        purify_E=("E" in config.leakage_correction),
#                        iterations=iterations
#                    )
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
#                elif config.leakage_correction=="mask_only":
#                    alm_out = hp.map2alm(np.array([0.*output_maps[0,:,c],output_maps[0,:,c],output_maps[1,:,c]]) * compsep_run["mask"],lmax=config.lmax, pol=True, **kwargs)
#            else:
            alm_out = hp.map2alm(
                [0.*output_maps[0,:,c], output_maps[0,:,c], output_maps[1,:,c]],
                lmax=config.lmax, pol=True, **kwargs
            )

            if (config.field_out in ["QU", "QU_E", "QU_B"]):
                output_maps[...,c] = hp.alm2map(alm_out, config.nside, lmax=config.lmax, pol=True, pixwin=True)[1:]
            elif config.field_out in ["E","B"]:
                output_maps[0,:,c] = hp.alm2map(alm_out[1] if config.field_out=="E" else alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
            elif config.field_out=="EB":
                output_maps[0,:,c] = hp.alm2map(alm_out[1], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)
                output_maps[1,:,c] = hp.alm2map(alm_out[2], config.nside, lmax=config.lmax, pol=False, pixwin=config.pixel_window_out)

        if config.field_out in ["E","B"]:
            output_maps = output_maps[0]

    # Apply mask if provided
    if "mask" in compsep_run:
        if output_maps.ndim == 3:
            output_maps[:,compsep_run["mask"] == 0.,:] = 0.
        elif output_maps.ndim == 2:
            output_maps[compsep_run["mask"] == 0.,:] = 0.

    return output_maps
        
def _get_fres_P_needlet_j(
    config: Configs,
    input_alms: np.ndarray,
    compsep_run: dict,
    b_ell: np.ndarray,
    nl_scale: int,
    **kwargs: Any
) -> np.ndarray:
    """
    Estimates the foreground residuals template from input foregrounds multifrequency templates using
    PILC component separation weights for a single needlet band.

    Parameters
    ----------
        config : Configs
            Configuration object containing the global settings. See 'get_residuals_template' for details.
        input_alms : np.ndarray
            Input foregrounds multifrequency templates in alm format for polarization field.
            The shape should be (n_freqs, (n_fields), n_alms, n_comps), with n_fields not None if both E and B fields are present.
        compsep_run : dict
            Dictionary containing the residuals estimates run parameters. See 'get_residuals_template' for details.
        b_ell : np.ndarray
            Needlet band to be used for the residuals estimation.
        nl_scale : int
            Scale of the needlet band, used to determine the weights filename.
        kwargs : dict
            Additional keywords for healpy map2alm function.
    
    Returns
    -------
        np.ndarray
            Estimate of foreground residuals for component separation in needlet domain for a single band.
            Shape is (n_pixels, n_comps).
    """

    nside_, lmax_ = config.nside, config.lmax
    good_channels_nl = _get_good_channels_nl(config, b_ell)

    # Load PILC weights
    weights_filename = os.path.join(config.path_outputs,
        compsep_run["compsep_path"],
        f"weights/weights_{compsep_run['field']}_{config.fwhm_out}acm_ns{nside_}_lmax{config.lmax}_nl{nl_scale}"
    )
    if compsep_run["nsim"] is not None:
        weights_filename += f"_{compsep_run['nsim']}"
    weights_filename += ".npy"
    w_pilc = np.load(weights_filename)

    input_maps_nl = np.zeros((good_channels_nl.shape[0], 2, 12 * nside_**2, input_alms.shape[-1]))
    # Convert alms to filtered Q/U maps
    for n, channel in enumerate(good_channels_nl):
        input_alms_j = np.zeros((2, hp.Alm.getsize(lmax_), input_alms.shape[-1]), dtype=complex)
        if input_alms.ndim == 4:
            for k in range(2):
                input_alms_j[k] = _needlet_filtering(input_alms[channel,k], b_ell, lmax_)
        elif input_alms.ndim == 3:
            if compsep_run["field_in_cs"] in ["QU_E", "E"]:
                input_alms_j[0] = _needlet_filtering(input_alms[channel], b_ell, lmax_)
            elif compsep_run["field_in_cs"] in ["QU_B", "B"]:
                input_alms_j[1] = _needlet_filtering(input_alms[channel], b_ell, lmax_)

        input_alms_j = np.ascontiguousarray(input_alms_j)

        for c in range(input_alms.shape[-1]):
            input_maps_nl[n,...,c] = hp.alm2map(
                np.array([0.*input_alms_j[0, :, c],input_alms_j[0, :, c],input_alms_j[1, :, c]]),
                nside_, lmax=lmax_, pol=True
            )[1:]

    # Apply PILC weights
    if w_pilc.ndim==1:
        output_maps_nl = np.einsum('i,ifjk->fjk', w_pilc, input_maps_nl)
    elif w_pilc.ndim==2:
        if w_pilc.shape[0] == input_maps_nl.shape[0]:
            output_maps_nl = np.einsum('ij,ifjk->fjk', w_pilc, input_maps_nl)
        elif w_pilc.shape[0] == 2:
            q = np.einsum('i,ijk->jk', w_pilc[0], input_maps_nl[:,0]) - np.einsum('i,ijk->jk', w_pilc[1], input_maps_nl[:,1])
            u = np.einsum('i,ijk->jk', w_pilc[1], input_maps_nl[:,0]) + np.einsum('i,ijk->jk', w_pilc[0], input_maps_nl[:,1])
            output_maps_nl = np.array([q, u])
    elif w_pilc.ndim==3:
        q = np.einsum('ij,ijk->jk', w_pilc[0], input_maps_nl[:,0]) - np.einsum('ij,ijk->jk', w_pilc[1], input_maps_nl[:,1])
        u = np.einsum('ij,ijk->jk', w_pilc[1], input_maps_nl[:,0]) + np.einsum('ij,ijk->jk', w_pilc[0], input_maps_nl[:,1])
        output_maps_nl = np.array([q, u])

    del input_maps_nl

    return output_maps_nl

def _get_fres_P_pixel(config: Configs, input_alms, compsep_run, **kwargs):
    """
    Estimates the foreground residuals template from input foregrounds multifrequency templates using 
    component separation PILC component separation weights in pixel domain.

    Parameters
    ----------
        config : Configs
            Configuration object containing the global settings. See 'get_residuals_template' for details.
        input_alms : np.ndarray
            Input foregrounds multifrequency templates in alm format for polarization field.
            The shape should be (n_freqs, (n_fields), n_alms, n_comps), with n_fields not None if both E and B fields are present.
        compsep_run : dict
            Dictionary containing the residuals estimates run parameters. See 'get_residuals_template' for details.
        kwargs : dict
            Additional keywords for healpy map2alm function.
    Returns
    -------
        np.ndarray
            Estimate of foreground residuals from PILC in pixel domain.
            Shape is (n_fields, n_pixels, n_comps) where n_fields depends on config.field_out.
    """
    good_channels = _get_good_channels_nl(config, np.ones(config.lmax + 1))

    input_maps = np.zeros((good_channels.shape[0], 2, 12 * config.nside**2, input_alms.shape[-1]))
    
    for n, channel in enumerate(good_channels):
        for c in range(input_alms.shape[-1]):
            if input_alms.ndim == 4:
                input_maps[n, ..., c] = hp.alm2map(
                    np.array([0. * input_alms[channel, 0, :, c],
                              input_alms[channel, 0, :, c],
                              input_alms[channel, 1, :, c]]),
                    config.nside, lmax=config.lmax, pol=True)[1:]
            elif input_alms.ndim == 3:
                if compsep_run["field_in_cs"] in ["QU_E", "E"]:
                    input_maps[n, ..., c] = hp.alm2map(
                        np.array([0. * input_alms[channel, :, c],
                                  input_alms[channel, :, c],
                                  0. * input_alms[channel, :, c]]),
                        config.nside, lmax=config.lmax, pol=True)[1:]
                elif compsep_run["field_in_cs"] in ["QU_B", "B"]:
                    input_maps[n, ..., c] = hp.alm2map(
                        np.array([0. * input_alms[channel, :, c],
                                  0. * input_alms[channel, :, c],
                                  input_alms[channel, :, c]]),
                        config.nside, lmax=config.lmax, pol=True)[1:]

    weights_filename = os.path.join(config.path_outputs, compsep_run["compsep_path"], "weights")
    weights_filename += f"/weights_{compsep_run['field']}_{config.fwhm_out}acm_ns{config.nside}_lmax{config.lmax}"
    if compsep_run["nsim"] is not None:
        weights_filename += f"_{compsep_run['nsim']}"
    weights_filename += ".npy"
    w_pilc = np.load(weights_filename)
    
    if w_pilc.ndim==1:
        output_maps = np.einsum('i,ifjk->fjk', w_pilc, input_maps)
    elif w_pilc.ndim==2:
        if w_pilc.shape[0] == input_maps.shape[0]:
            output_maps = np.einsum('ij,ifjk->fjk', w_pilc, input_maps)
        elif w_pilc.shape[0] == 2:
            q = np.einsum('i,ijk->jk', w_pilc[0], input_maps[:,0]) - np.einsum('i,ijk->jk', w_pilc[1], input_maps[:,1])
            u = np.einsum('i,ijk->jk', w_pilc[1], input_maps[:,0]) + np.einsum('i,ijk->jk', w_pilc[0], input_maps[:,1])
            output_maps = np.array([q, u])
    elif w_pilc.ndim==3:
        q = np.einsum('ij,ijk->jk', w_pilc[0], input_maps[:,0]) - np.einsum('ij,ijk->jk', w_pilc[1], input_maps[:,1])
        u = np.einsum('ij,ijk->jk', w_pilc[1], input_maps[:,0]) + np.einsum('ij,ijk->jk', w_pilc[0], input_maps[:,1])
        output_maps = np.array([q, u])

    process_outputs = (
        ((config.field_out in ["QU", "QU_E", "QU_B"]) and config.pixel_window_out) or
        (config.field_out in ["E", "B", "EB"])
    )

    if process_outputs:
        for c in range(output_maps.shape[-1]):
#            if "mask" in compsep_run and config.mask_type == "observed_patch" and config.leakage_correction is not None:
#                if "_purify" in config.leakage_correction:
#                    alm_out = purify_master(
#                        output_maps[..., c], compsep_run["mask"], config.lmax,
#                        purify_E=("E" in config.leakage_correction)
#                    )
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
#                elif "_recycling" in config.leakage_correction:
#                    iterations = (
#                        int(re.search(r'iterations(\d+)', config.leakage_correction).group(1))
#                        if "_iterations" in config.leakage_correction else 0
#                    )
#                    alm_out = purify_recycling(
#                        output_maps[..., c], output_maps[..., 0],
#                        np.ceil(compsep_run["mask"]), config.lmax,
#                        purify_E=("E" in config.leakage_correction),
#                        iterations=iterations
#                    )
#                    alm_out = np.concatenate([(0.*alm_out[0])[np.newaxis],alm_out], axis=0)
                #elif config.leakage_correction=="mask_only":
                #    alm_out = hp.map2alm(np.array([0.*output_maps[0,:,c],output_maps[0,:,c],output_maps[1,:,c]]) * compsep_run["mask"],lmax=config.lmax, pol=True, **kwargs)
#            else:
            alm_out = hp.map2alm(
                [0.*output_maps[0,:,c], output_maps[0,:,c], output_maps[1,:,c]],
                lmax=config.lmax, pol=True, **kwargs
            )

            if (config.field_out in ["QU", "QU_E", "QU_B"]):
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

__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
]

