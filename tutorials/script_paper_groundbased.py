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
    'experiment': "SO_SAT", 
    'units': "uK_CMB",
    'coordinates': "C", 
    'instrument':
    {"frequency": [27.0,    39.0,   93.0,   145.0,  225.0,  280.0],
    "depth_P": [46.,    28.,   3.5,    4.4,    8.4,    21.],
    "fwhm": [91.0,    63.0,   30.0,   17.0,   11.0,   9.0],
    "beams": "gaussian",
    "ell_knee":   [[15, 15, 25, 25, 35, 40],[15, 15, 25, 25, 35, 40]],
    "alpha_knee": [-2.4, -2.4, -2.5, -3.,-3.,-3.],
    "path_hits_maps": os.path.join(root_path, "utils", "norm_nHits_SA_35FOV_ns512.fits")
    }
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
    'mask_observations': os.path.join(root_path, "utils", "norm_nHits_SA_35FOV_ns512.fits"),
    'mask_covariance': os.path.join(root_path, "utils", "norm_nHits_SA_35FOV_ns512.fits"), #
    'leakage_correction': None,
    'save_compsep_products': True, 
    'return_compsep_products': False, #
    'path_outputs': f"outputs/{config.experiment}/{''.join(config.foreground_models)}",
    'field_in': "TQU", 
    'field_out': "TQU", # 
}

config_values.update(config_common_cs)
config = Configs(config=config_values)

# Component separation run for TEB
config_run_TEB = {
    'compsep': [
         {
    # ILC in pixel space
        {
    'method': "ilc",
    'domain': "pixel",
    'ilc_bias': 0.001,
    'reduce_ilc_bias': False,
    'cov_noise_debias': 0.,
    'load_noise_covariance': False,
    'component_out': 'cmb',
        },
    # ILC in needlet space
    {'method': "ilc",
    'domain': "needlet", # needlet domain
    'ilc_bias': 0.001,
    'needlet_config':
     [{'needlet_windows': "mexican"},
       {'width': 1.3},
       {'merging_needlets': [0, 16, 19, 22,40]}],
    'reduce_ilc_bias': False,
    'b_squared': False,
    'adapt_nside': False,
    'save_needlets': True,
    'save_weights': True,
    'cov_noise_debias': [0.,0.,0.,0.],
    'load_noise_covariance': False,
    'component_out': 'cmb',
    },
    # Generalized ILC for foreground reconstruction
    {'method': "gilc",
    'domain': "needlet",
    'needlet_config':
      [{'needlet_windows': "mexican"},
       {'width': 1.3},
       {'merging_needlets': [0, 16, 19, 22, 40]}], #[0, 14, 17, 19, 40]}],
    'ilc_bias': 0.001,
    'b_squared': False,
    'adapt_nside': False,
    # 'cmb_nuisance': True,
    'depro_cmb': [None, None, None], #[None,0.,0.,0.]
    'm_bias': [0,0,0,0],
    'channels_out': [0,2,5],
    'save_needlets': True,
    'cov_noise_debias': [0.,0.,0.,0.],
    'load_nuisance_covariance': False},
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
    'generate_input_foregrounds': False, 
    'generate_input_noise': False, #
    'generate_input_cmb': False, #
    'field_out': "QU", # 
    'compsep': [
     # constrained ILC with deprojection of all 0th order moments
   {'method': "cilc",
   'domain': "needlet",
   'ilc_bias': 0.001,
   'needlet_config':
     [{'needlet_windows': "mexican"},
      {'width': 1.3},
      {'merging_needlets': [0, 16, 19, 22, 40]}], #[0, 14, 17, 19, 40]}],
   'component_out': 'cmb',
   'constraints':
     [
      {'components': ["0d","0s"]}, # 
         {'beta_d': 1.54}, 
         {'T_d': 20.}, 
         {'beta_s': -3.}, 
       {'deprojection': [0., 0.]} # or [[0., 0., 0., 0., 0.], [0.1, 0.1, 0.1, 0.1, 0.1], ...]
     ],
   'adapt_nside': False,
   'save_weights': True,
   'cov_noise_debias': [0.,0.,0.,0.],}  
    ]}

config_values.update(config_run_EB)
config = Configs(config=config_values)

# Running compsep pipelines for E, B fields
if parallelize:
    for idx in idxs_mpi:
        run_compsep(config, idx)
else:
    for nsim in range(config.nsim_start,config.nsim_start+config.nsims):
        run_compsep(config, nsim)

# Perform NILC component separation with different EB-leakage correction techniques

# Recycling technique with three iterations
config_run_leak = {
    'leakage_correction': "B_recycling_iterations3",
    'compsep': [
    {'method': "ilc",
    'domain': "needlet", # needlet domain
    'ilc_bias': 0.001,
    'needlet_config':
     [{'needlet_windows': "mexican"},
       {'width': 1.3},
       {'merging_needlets': [0, 16, 19, 22,40]}],
    'reduce_ilc_bias': False,
    'b_squared': False,
    'adapt_nside': False,
    'save_needlets': True,
    'save_weights': True,
    'cov_noise_debias': [0.,0.,0.,0.],
    'load_noise_covariance': False,
    'component_out': 'cmb',
    },]}
config_values.update(config_run_leak)
config = Configs(config=config_values)
    
# Running compsep pipelines for B field
if parallelize:
    for idx in idxs_mpi:
        run_compsep(config, idx)
else:
    for nsim in range(config.nsim_start,config.nsim_start+config.nsims):
        run_compsep(config, nsim)

# Recycling technique with three iterations
config_run_leak = {
    'leakage_correction': "B_recycling_inpainting",
    'compsep': [
    {'method': "ilc",
    'domain': "needlet", # needlet domain
    'ilc_bias': 0.001,
    'needlet_config':
     [{'needlet_windows': "mexican"},
       {'width': 1.3},
       {'merging_needlets': [0, 16, 19, 22,40]}],
    'reduce_ilc_bias': False,
    'b_squared': False,
    'adapt_nside': False,
    'save_needlets': True,
    'save_weights': True,
    'cov_noise_debias': [0.,0.,0.,0.],
    'load_noise_covariance': False,
    'component_out': 'cmb',
    },]}
config_values.update(config_run_leak)
config = Configs(config=config_values)
    
# Running compsep pipelines for B field
if parallelize:
    for idx in idxs_mpi:
        run_compsep(config, idx)
else:
    for nsim in range(config.nsim_start,config.nsim_start+config.nsims):
        run_compsep(config, nsim)

# Recycling technique with three iterations
config_run_leak = {
    'leakage_correction': "B_inpainting_iterations3",
    'compsep': [
    {'method': "ilc",
    'domain': "needlet", # needlet domain
    'ilc_bias': 0.001,
    'needlet_config':
     [{'needlet_windows': "mexican"},
       {'width': 1.3},
       {'merging_needlets': [0, 16, 19, 22,40]}],
    'reduce_ilc_bias': False,
    'b_squared': False,
    'adapt_nside': False,
    'save_needlets': True,
    'save_weights': True,
    'cov_noise_debias': [0.,0.,0.,0.],
    'load_noise_covariance': False,
    'component_out': 'cmb',
    },]}
config_values.update(config_run_leak)
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
    'delta_ell': 10, 
    'spectra_comp': 'namaster',
    'return_Dell': False, 
    'return_spectra': False, 
    'save_spectra': True,  
    'save_mask': True,
}
config_values.update(config_cls_values)
config = Configs(config=config_values)

# Power spectrum calculation from temperature selected outputs

config_cls_run_EB = {
    'field_cls_out': ["EE","BB"],
    'compute_spectra': [  
    {
    'path_method': "ilc_pixel_bias0.001/cmb_reconstruction", #cosine_0_100_200_300_500_700", #
    'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': "from_fits",
    'mask_path': os.path.join(root_path, "utils", "norm_nHits_SA_35FOV_ns512.fits"), #
    'field_out': 'TQU',
    'apodize_mask': "C1",
    'apodize_scale': 10.,
    'nmt_purify_B': True,
    'nmt_purify_E': False,
    },  
    {
    'path_method': "ilc_needlet_bias0.001/mexican_B1.3_j0j15_j16j18_j19j21_j22j39/cmb_reconstruction", 
    'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': "from_fits",
    'mask_path': os.path.join(root_path, "utils", "norm_nHits_SA_35FOV_ns512.fits"), #
    'field_out': 'TQU',
    'apodize_mask': "C1",
    'apodize_scale': 10.,
    'nmt_purify_B': True,
    'nmt_purify_E': False,
    },  
     {'path_method': "cilc_needlet_bias0.001/mexican_B1.3_j0j15_j16j18_j19j21_j22j39/cmb_reconstruction_0d0s",
     'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': "from_fits",
    'mask_path': os.path.join(root_path, "utils", "norm_nHits_SA_35FOV_ns512.fits"), #
    'field_out': 'QU',
    'apodize_mask': "C1",
    'apodize_scale': 10.,
    'nmt_purify_B': True,
    'nmt_purify_E': False,},
    {'path_method': "gilc_needlet_bias0.001_nuis_noise+cmb/mexican_B1.3_j0j15_j16j18_j19j21_j22j39",
    'components_for_cls': ["noise_residuals_27.0GHz", "fgds_residuals_27.0GHz", "output_cmb_27.0GHz", "noise_residuals_93.0bGHz", "fgds_residuals_93.0bGHz", "output_cmb_93.0bGHz", "noise_residuals_280.0GHz", "fgds_residuals_280.0GHz", "output_cmb_280.0GHz"],
    'mask_type': "from_fits",
    'mask_path': os.path.join(root_path, "utils", "norm_nHits_SA_35FOV_ns512.fits"), #
    'field_out': 'QU',
    'apodize_mask': "C1",
    'apodize_scale': 10.,
    'nmt_purify_B': True,
    'nmt_purify_E': False,
    },]}

config_values.update(config_cls_run_EB)
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
    'path_method': "ilc_needlet_bias0.001_Brecycling_iters3/mexican_B1.3_j0j15_j16j18_j19j21_j22j39/cmb_reconstruction", 
    'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': "from_fits",
    'mask_path': os.path.join(root_path, "utils", "norm_nHits_SA_35FOV_ns512.fits"), #
    'field_out': 'TQU',
    'apodize_mask': "C1",
    'apodize_scale': 10.,
    'nmt_purify_B': True,
    'nmt_purify_E': False,
    },  
    {
    'path_method': "ilc_needlet_bias0.001_Brecycling_inpainting/mexican_B1.3_j0j15_j16j18_j19j21_j22j39/cmb_reconstruction", 
    'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': "from_fits",
    'mask_path': os.path.join(root_path, "utils", "norm_nHits_SA_35FOV_ns512.fits"), #
    'field_out': 'TQU',
    'apodize_mask': "C1",
    'apodize_scale': 10.,
    'nmt_purify_B': True,
    'nmt_purify_E': False,
    },  
    {
    'path_method': "ilc_needlet_bias0.001_Binpainting_iters3/mexican_B1.3_j0j15_j16j18_j19j21_j22j39/cmb_reconstruction", 
    'components_for_cls': ["output_total",["output_total_split1","output_total_split2"], "noise_residuals", "fgds_residuals", "output_cmb"],
    'mask_type': "from_fits",
    'mask_path': os.path.join(root_path, "utils", "norm_nHits_SA_35FOV_ns512.fits"), #
    'field_out': 'TQU',
    'apodize_mask': "C1",
    'apodize_scale': 10.,
    'nmt_purify_B': True,
    'nmt_purify_E': False,
    }, ]}

config_values.update(config_cls_run_B)
config = Configs(config=config_values)

# Computing spectra
if parallelize:
    for idx in idxs_mpi:
        get_spectra(config, idx)
else:
    for nsim in range(config.nsim_start,config.nsim_start+config.nsims):
        get_spectra(config, nsim)
