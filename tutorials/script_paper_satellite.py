import os 
import numpy as np
import healpy as hp
import broom
from types import SimpleNamespace
import pymaster as nmt 
import re
from broom import (
    Configs,
    get_params,
    component_separation,get_input_data, estimate_residuals, _combine_products, _compute_spectra, combine_with_weights, get_nuisance_covariance)

from broom.clusters import get_mcilc_tracers
from broom.routines import _format_nsim

root_path = os.path.dirname(os.path.abspath(broom.__file__))
import sys

parallelize = True

if parallelize:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

def run_compsep(config: Configs, nsim):
    nsim = _format_nsim(nsim)
    
    data = get_input_data(config, nsim = nsim, iter=8, use_weights=True)

    component_separation(config, data, nsim=nsim, iter=8, use_pixel_weights=True)

    # estimate_residuals(config, nsim=nsim)
    
    return None

def get_spectra(config: Configs, nsim):
    nsim = _format_nsim(nsim)
    
    _compute_spectra_(config, nsim = nsim)
    
    return None

# General parameters 
config_values = {
    'lmin': 0, 
    'lmin_in': 0,     
    'nside': 512, 
    'nside_in': 512,
    'lmax': 1024, 
    'lmax_in': 1024, 
    'data_type': "maps", 
    'verbose': True,  
    'nsim_start': 0,
    'nsims': 10,
    'foreground_models': ["d1","s1","co3","a1","f1","tsz1","cib1","ksz1"], 
    'experiments_file': root_path + "/utils/experiments.yaml",
    'experiment': "LiteBIRD_PTEP", 
    'units': "uK_CMB",
    'coordinates': "G", 
    'instrument':
    {"frequency": [40.0,    50.0,   60.0,   68.0,   68.0,   78.0,   78.0,   89.0,   89.0,   100.0,   100.0,  119.0,  119.0,  140.0,  140.0,  166.0,  195.0,  195.0,  235.0,  280.0,  337.0,  402.0],
    "depth_P": [37.42,    33.46,     21.31,   19.91,   31.77,   15.55,   19.13,    12.28,    28.77,    10.34,    8.48,    7.69,    5.70,   7.25,   6.38, 5.57, 7.05, 10.50, 10.79, 13.80, 21.95, 47.45],
    "fwhm": [70.5,    58.5,   51.1,   41.6,   47.1,   36.9,   43.8,   33.0,   41.5,   30.2,   37.8,   26.3,   33.6,   23.7,   30.8, 28.9, 28.0, 28.6, 24.7, 22.5, 20.9, 17.9],
    "beams": "gaussian",}
}

# Initializing the configuration class
config = Configs(config=config_values)

# Parameters for input simulations
config_inputs_values = {
    'generate_input_foregrounds': True, 
    'return_fgd_components': False,
    'bandpass_integrate': False,
    'generate_input_noise': True, #
    'seed_noise': None,
    'data_splits': True,
    'only_splits': False,
    'generate_input_cmb': True, #
    'seed_cmb': False,
    'cls_cmb_path': root_path + "/utils/Cls_Planck2018_lensed_r0.fits",
    'cls_cmb_new_ordered': True,
    'generate_input_data': True, #
    'save_inputs': True, # 
    'pixel_window_in': True,
    'data_path': f"inputs/{config.experiment}/total/total_{config.data_type}_ns{config.nside_in}_lmax{config.lmax_in}",
    'fgds_path': f"inputs/{config.experiment}/foregrounds/{''.join(config.foreground_models)}/foregrounds_alms_ns{config.nside_in}_lmax{config.lmax_in}",
    'cmb_path': f"inputs/{config.experiment}/cmb/cmb_{config.data_type}_ns{config.nside_in}_lmax{config.lmax_in}",
    'noise_path': f"inputs/{config.experiment}/noise/noise_{config.data_type}_ns{config.nside_in}_lmax{config.lmax_in}",
}

config_values.update(config_inputs_values)
config = Configs(config=config_values)

# Common parameter for component separation (and power spectra correction).
config_common_cs = {
    'fwhm_out': 30., 
    'bring_to_common_resolution': True, 
    'pixel_window_out': True,
    'mask_observations': None,
    'mask_covariance': None, #
    'leakage_correction': None,
    'save_compsep_products': True, 
    'return_compsep_products': False, #
    'path_outputs': f"outputs/{config.experiment}/{''.join(config.foreground_models)}",
    'field_in': "TQU", 
    'field_out': "TEB", # 
}

config_values.update(config_common_cs)
config = Configs(config=config_values)

# Component separation run for TEB
config_run_TEB = {
    'compsep': [
         {
    # ILC in pixel space
     'method': "ilc",
     'domain': "pixel",
     'ilc_bias': 0.001,
     'reduce_ilc_bias': False,
     'cov_noise_debias': 0.,
     'load_noise_covariance': False,
     'component_out': 'cmb',
     },
    # ILC in needlet domain
     {
     'method': "ilc",
     'domain': "needlet", # needlet domain
     'ilc_bias': 0.001,
     'needlet_config':
      [{'needlet_windows': "mexican"},
        {'width': 1.3},
        {'merging_needlets': [0, 16, 19, 22, 25,28,40]}],
     'reduce_ilc_bias': False,
     'b_squared': False,
     'adapt_nside': True,
     'save_needlets': True,
     'save_weights': True,
     'cov_noise_debias': [0.,0.,0.,0.],
     'load_noise_covariance': False,
     'component_out': 'cmb',},
    # Generalized ILC for foreground reconstruction
   {'method': "gilc",
   'domain': "needlet",
   'needlet_config':
     [{'needlet_windows': "mexican"},
      {'width': 1.3},
      {'merging_needlets': [0, 16, 19, 22, 25,28,40]}], #[0, 14, 17, 19, 40]}],
   'ilc_bias': 0.005,
   'b_squared': False,
   'adapt_nside': True,
    'cmb_nuisance': True,
   'depro_cmb': [None, None, None], #[None,0.,0.,0.]
   'm_bias': [0,0,0,0],
   'channels_out': [0, 12, 21],
   'save_needlets': True,
   'cov_noise_debias': [0.,0.,0.,0.],
   'load_nuisance_covariance': False},
    # Foreground complexity diagnostic
    {'method': "fgd_diagnostic",
    'domain': "needlet",
    'needlet_config':
      [{'needlet_windows': "mexican"},
       {'width': 1.3},
       {'merging_needlets': [0, 16, 19, 22, 25,28,40]}], #[0, 14, 17, 19, 40]}],
    'ilc_bias': 0.5,
    'b_squared': False,   
    'adapt_nside': True,
    'cmb_nuisance': True,
    'save_needlets': True,
    'cov_noise_debias': [0.,0.,0.,0.]}
]}
config_values.update(config_run_TEB)
config = Configs(config=config_values)

# Running compsep pipelines for T, E, B fields
if parallelize:
    complete_idxs = np.arange(config.nsim_start,config.nsim_start+config.nsims)
    idxs_mpi = None
    if rank == 0:
        idxs_mpi = complete_idxs[:]
        idxs_mpi = np.array_split(idxs_mpi, size)
    idxs_mpi = comm.scatter(idxs_mpi, root=0)
    for idx in idxs_mpi:
        run_compsep(config, idx)
else:
    for nsim in range(config.nsim_start,config.nsim_start+config.nsims):
        run_compsep(config, nsim)

# Component separation just for E and B modes
config_run_EB = {
    # At this stage, input data have been generated and saved.
    # No need to generate new sims.
    # Setting them to false will allow to just load them from disk.
    'generate_input_foregrounds': False, 
    'generate_input_noise': False, #
    'generate_input_cmb': False, #
    'field_out': "EB", # 
    'compsep': [
     # constrained ILC with deprojection of all 0th and 1st order moments
    {'method': "cilc",
    'domain': "needlet",
    'ilc_bias': 0.001,
#    # Needlet configuration
    'needlet_config':
      [{'needlet_windows': "mexican"},
       {'width': 1.3},
       {'merging_needlets': [0, 16, 19, 22, 25,28,40]}], #[0, 14, 17, 19, 40]}],
    'component_out': 'cmb',
    'constraints':
      [
       {'components': ["0d","0s","1bd","1bs","1Td"]},
          {'beta_d': 1.54}, # or [1.54, 1.5, 1.6, 1.55] for each needlet band
         {'T_d': 20.}, # or [20., 19., 21., 20.] for each needlet band
          {'beta_s': -3.}, # or [-3., -2.5, -3.5, -3.] for each needlet band
        {'deprojection': [0., 0., 0., 0., 0.]} # or [[0., 0., 0., 0., 0.], [0.1, 0.1, 0.1, 0.1, 0.1], ...]
      ],
    'save_weights': True,
    'cov_noise_debias': [0.,0.,0.,0.],},
    # constrained ILC with deprojection of all 0th order moments and 1st in beta_dust
    {'method': "cilc",
    'domain': "needlet",
    'ilc_bias': 0.001,
    'needlet_config':
      [{'needlet_windows': "mexican"},
       {'width': 1.3},
       {'merging_needlets': [0, 16, 19, 22, 25,28,40]}], #[0, 14, 17, 19, 40]}],
    'component_out': 'cmb',
    'constraints':
      [
       {'components': ["0d","0s","1bd"]}, # 
          {'beta_d': 1.54}, 
          {'T_d': 20.}, 
          {'beta_s': -3.}, 
        {'deprojection': [0., 0.,0.]} # or [[0., 0., 0., 0., 0.], [0.1, 0.1, 0.1, 0.1, 0.1], ...]
      ],
    'save_weights': True,
    'cov_noise_debias': [0.,0.,0.,0.],},
    # Polarization ILC with real weights (for complex weights "pilc")
    {'method': "prilc",
    'domain': "needlet", # needlet domain
    'ilc_bias': 0.001,
    'needlet_config':
     [{'needlet_windows': "mexican"},
       {'width': 1.3},
       {'merging_needlets': [0, 16, 19, 22, 25,28,40]}], 
    'reduce_ilc_bias': False,
    'b_squared': False,
    'adapt_nside': True,
    'save_needlets': True,
    'save_weights': True,
    'cov_noise_debias': [0.,0.,0.,0.],
    'load_noise_covariance': False,
    'component_out': 'cmb',
    'nu_ref_s_out': 40.,
    'beta_d_out': 1.54,
    'T_d_out': 20.}]}

config_values.update(config_run_EB)
config = Configs(config=config_values)

# Running compsep pipelines for E, B fields
if parallelize:
    for idx in idxs_mpi:
        run_compsep(config, idx)
else:
    for nsim in range(config.nsim_start,config.nsim_start+config.nsims):
        run_compsep(config, nsim)

# Running tracers generation for sky partition for realistic MC-ILC 
# (needed just once per instrument and foreground model)

config_clusters = {
    'real_mc_tracers': [{
        'channels_tracers': [20,12],
        'path_tracers': config.path_outputs + "/tracers",
        'tracers_inputs_path': f"mc_tracers_inputs/{config.experiment}"
}]}
config_values.update(config_clusters)
config = Configs(config=config_values)

if parallelize:
    if rank == 0:
        get_mcilc_tracers(config)
    comm.Barrier()
else:
    get_mcilc_tracers(config)

# Component separation just for B modes
config_run_B = {
    'field_out': "B", # 
    'compsep': [
    # MC-ILC with ideal clusters
    {'method': "mcilc",
    'domain': "needlet",
    'needlet_config':
      [{'needlet_windows': "mexican"},
       {'width': 1.3},
       {'merging_needlets': [0, 16, 19, 22, 25,28,40]}], #[0, 14, 17, 19, 40]}],
    'reduce_mcilc_bias': False,
    'mc_type': "cea_ideal", # Ideal clusters
    'channels_tracers': [20,12], # 
    'n_patches': 50,
    'save_patches': True},
    # MC-ILC with realistic patches obtained from tracers computed in step above
    {'method': "mc_ilc",
    'domain': "needlet",
    'needlet_config':
      [{'needlet_windows': "mexican"},
       {'width': 1.3},
       {'merging_needlets': [0, 16, 19, 22, 25,28,40]}], 
    'reduce_mcilc_bias': False,
    'mc_type': "cea_real", # "cea_ideal", "rp_ideal", "cea_real", "rp_real"
    'channels_tracers': [20,12], #
    'ilc_bias': 0.001,
    'n_patches': 50,
    # path where the MC-ILC tracers are stored. Used only if mc_type is "cea_real" or "rp_real".
    'path_tracers': config.path_outputs + "/tracers", 
    'special_nls': [0],
    'save_patches': True,}]}
config_values.update(config_run_B)
config = Configs(config=config_values)
    
# Running compsep pipelines for B field
if parallelize:
    for idx in idxs_mpi:
        run_compsep(config, idx)
else:
    for nsim in range(config.nsim_start,config.nsim_start+config.nsims):
        run_compsep(config, nsim)

# Power spectrum calculation general parameters

config_cls_values = {
    'delta_ell': 5, 
    'spectra_comp': 'namaster',
    'return_Dell': False, 
    'return_spectra': False, 
    'save_spectra': True,  
    'save_mask': True,
}
config_values.update(config_cls_values)
config = Configs(config=config_values)

# Power spectrum calculation from temperature selected outputs

config_cls_run_T = {
    'field_cls_out': ["TT"],
    'compute_spectra': [
     {
     'path_method': "ilc_pixel_bias0.001/cmb_reconstruction", 
     'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
     'mask_type': 'GAL80',
     'mask_path': None, #"",
     'field_out': 'TEB',
     'apodize_mask': "C1",
     'apodize_scale': 5.,
     },   
    {
    'path_method': "ilc_needlet_bias0.001/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39/cmb_reconstruction",
    'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': 'GAL80',
    'mask_path': None, #"",
    'field_out': 'TEB',
    # 'path_outputs': "outputs/LiteBIRD_PTEP/d1s1",
    'apodize_mask': "C1",
    'apodize_scale': 5.,
    'nmt_purify_B': False,
    'nmt_purify_E': False,
    }, 
    {
    'path_method': "gilc_needlet_bias0.005_nuis_noise+cmb/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39",
    'components_for_cls': ["noise_residuals_40.0GHz", "fgds_residuals_40.0GHz", "output_cmb_40.0GHz", "noise_residuals_119.0bGHz", "fgds_residuals_119.0bGHz", "output_cmb_119.0bGHz", "noise_residuals_402.0GHz", "fgds_residuals_402.0GHz", "output_cmb_402.0GHz"],
    'mask_type': 'GAL80',
    'mask_path': None, #"",
    'field_out': 'TEB',
    # 'path_outputs': "outputs/LiteBIRD_PTEP/d1s1",
    'apodize_mask': "C1",
    'apodize_scale': 5.,
    'nmt_purify_B': False,
    'nmt_purify_E': False,
    }, 
    ]}

config_values.update(config_cls_run_T)
config = Configs(config=config_values)

# Computing spectra
if parallelize:
    for idx in idxs_mpi:
        get_spectra(config, idx)
else:
    for nsim in range(config.nsim_start,config.nsim_start+config.nsims):
        get_spectra(config, nsim)


config_cls_run_E = {
    'field_cls_out': ["EE"],
    'compute_spectra': [  
    {
    'path_method': "ilc_pixel_bias0.001/cmb_reconstruction", #cosine_0_100_200_300_500_700", #
    'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': 'GAL70',
    'mask_path': None, #"",
    'field_out': 'TEB',
    # 'path_outputs': "outputs/LiteBIRD_PTEP/d1s1",
    'apodize_mask': "C1",
    'apodize_scale': 5.,
    'nmt_purify_B': False,
    'nmt_purify_E': False,
    },  
    {
    'path_method': "ilc_needlet_bias0.001/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39/cmb_reconstruction", 
    'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': 'GAL70',
    'mask_path': None, #"",
    'field_out': 'TEB',
    'apodize_mask': "C1",
    'apodize_scale': 5.,
    'nmt_purify_B': False,
    'nmt_purify_E': False,
    },  
    {
    'path_method': "prilc_needlet_bias0.001/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39_nlsquared/cmb_reconstruction", 
    'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': 'GAL70',
    'mask_path': None, #"",
    'field_out': 'EB',
    # 'path_outputs': "outputs/LiteBIRD_PTEP/d1s1",
    'apodize_mask': "C1",
    'apodize_scale': 5.,
    'nmt_purify_B': False,
    'nmt_purify_E': False,},   
     {'path_method': "cilc_needlet_bias0.001/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39/cmb_reconstruction_0d0s1bd",
     'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
     'mask_type': 'GAL70',
     'mask_path': None, #"",
     'field_out': 'EB',
     # 'path_outputs': "outputs/LiteBIRD_PTEP/d1s1",
     'apodize_mask': "C1",
     'apodize_scale': 5.,
     'nmt_purify_B': True,
     'nmt_purify_E': False,},
     {'path_method': "cilc_needlet_bias0.001/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39/cmb_reconstruction_0d0s1bd1bs1Td",
     'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
     'mask_type': 'GAL70',
     'mask_path': None, #"",
     'field_out': 'EB',
     'apodize_mask': "C1",
     'apodize_scale': 5.,
     'nmt_purify_B': True,
     'nmt_purify_E': False,},
    {'path_method': "gilc_needlet_bias0.005_nuis_noise+cmb/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39",
    'components_for_cls': ["noise_residuals_40.0GHz", "fgds_residuals_40.0GHz", "output_cmb_40.0GHz", "noise_residuals_119.0bGHz", "fgds_residuals_119.0bGHz", "output_cmb_119.0bGHz", "noise_residuals_402.0GHz", "fgds_residuals_402.0GHz", "output_cmb_402.0GHz"],
    'mask_type': 'GAL70',
    'mask_path': None, #"",
    'field_out': 'TEB',
    # 'path_outputs': "outputs/LiteBIRD_PTEP/d1s1",
    'apodize_mask': "C1",
    'apodize_scale': 5.,
    'nmt_purify_B': False,
    'nmt_purify_E': False,
    },]}

config_values.update(config_cls_run_E)
config = Configs(config=config_values)

# Computing spectra
if parallelize:
    for idx in idxs_mpi:
        get_spectra(config, idx)
else:
    for nsim in range(config.nsim_start,config.nsim_start+config.nsims):
        get_spectra(config, nsim)

config_cls_run_B = {
    'field_cls_out': ["BB"],
    'compute_spectra': [  
    {
    'path_method': "ilc_needlet_bias0.001/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39/cmb_reconstruction", 
    'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': 'GAL60',
    'mask_path': None, #"",
    'field_out': 'TEB',
    'apodize_mask': "C1",
    'apodize_scale': 5.,
    'nmt_purify_B': False,
    'nmt_purify_E': False,
    },  
    {
    'path_method': "prilc_needlet_bias0.001/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39_nlsquared/cmb_reconstruction", 
    'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': 'GAL60',
    'mask_path': None, #"",
    'field_out': 'EB',
    # 'path_outputs': "outputs/LiteBIRD_PTEP/d1s1",
    'apodize_mask': "C1",
    'apodize_scale': 5.,
    'nmt_purify_B': False,
    'nmt_purify_E': False,},   
     {'path_method': "cilc_needlet_bias0.001/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39/cmb_reconstruction_0d0s1bd",
     'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
     'mask_type': 'GAL60',
     'mask_path': None, #"",
     'field_out': 'TEB',
     # 'path_outputs': "outputs/LiteBIRD_PTEP/d1s1",
     'apodize_mask': "C1",
     'apodize_scale': 5.,
     'nmt_purify_B': True,
     'nmt_purify_E': False,},
     {'path_method': "cilc_needlet_bias0.001/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39/cmb_reconstruction_0d0s1bd1bs1Td",
     'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
     'mask_type': 'GAL60',
     'mask_path': None, #"",
     'field_out': 'EB',
     'apodize_mask': "C1",
     'apodize_scale': 5.,
     'nmt_purify_B': True,
     'nmt_purify_E': False,},
    {
     'path_method': "mcilc_needlet/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39/cea_ideal_337.0GHz_119.0bGHz_50patches/cmb_reconstruction",
     'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
     'mask_type': 'GAL60',
     'mask_path': None, #"",
     'field_out': 'B',
     # 'path_outputs': "outputs/LiteBIRD_PTEP_GAL70_old/d1s1",
     'apodize_mask': "C1",
     'apodize_scale': 8.,
     'nmt_purify_B': False,
     'nmt_purify_E': False,},   
     {
     'path_method': "mc_ilc_needlet_bias0.001_nls0/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39/cea_real_337.0GHz_119.0bGHz_50patches/cmb_reconstruction",
     'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
     'mask_type': 'GAL60',
     'mask_path': None, #"",
     'field_out': 'B',
     'apodize_mask': "C1",
     'apodize_scale': 8.,
     'nmt_purify_B': False,
     'nmt_purify_E': False,},
    {'path_method': "gilc_needlet_bias0.005_nuis_noise+cmb/mexican_B1.3_j0j15_j16j18_j19j21_j22j24_j25j27_j28j39",
    'components_for_cls': ["noise_residuals_40.0GHz", "fgds_residuals_40.0GHz", "output_cmb_40.0GHz", "noise_residuals_119.0bGHz", "fgds_residuals_119.0bGHz", "output_cmb_119.0bGHz", "noise_residuals_402.0GHz", "fgds_residuals_402.0GHz", "output_cmb_402.0GHz"],
    'mask_type': 'GAL60',
    'mask_path': None, #"",
    'field_out': 'TEB',
    # 'path_outputs': "outputs/LiteBIRD_PTEP/d1s1",
    'apodize_mask': "C1",
    'apodize_scale': 5.,
    'nmt_purify_B': False,
    'nmt_purify_E': False,
    },]}

config_values.update(config_cls_run_B)
config = Configs(config=config_values)

# Computing spectra
if parallelize:
    for idx in idxs_mpi:
        get_spectra(config, idx)
else:
    for nsim in range(config.nsim_start,config.nsim_start+config.nsims):
        get_spectra(config, nsim)
