import os
from dataclasses import dataclass, field
import yaml
from typing import Dict, Any, Optional
import string
import numpy as np
import healpy as hp
import sys
import argparse
import yaml


def join(loader: yaml.Loader, node: yaml.Node) -> str:
    """
    Custom YAML tag to join sequences into a single string.

    Parameters
    ----------
        loader : yaml.Loader
            The YAML loader instance.
        node : yaml.Node
            The YAML node to process.

    Returns
    -------
        str
            A string formed by joining the sequence elements.
    """
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments, especially the configuration file path.

    Parameters
    ----------
        None

    Returns
    -------
        argparse.Namespace
            The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Parse arguments and return parser object"
    )

    parser.add_argument(
        "-C",
        "--config_path",
        type=str,
        default="configs/test_config.yaml",
        help="Path to the configuration file. If not provided, takes the default path.",
    )

    return parser.parse_args()

def get_params(config_path: Optional[str] = None) -> "Configs":
    """
    Get parameters from a YAML configuration file.

    Parameters
    ----------
        config_path : str, optional
            Path to the configuration file. If not provided, uses the default path.

    Returns
    -------
        Configs
            An instance of the Configs class containing the configuration parameters.
    """
    
    yaml.add_constructor("!join", join)
    if config_path is None:
        parsed_args = parse_args()
        config_path = os.path.abspath(parsed_args.config_path)
    return Configs(config_path=config_path)

@dataclass
class InstrumentConfig:
    #frequency: list = field(default_factory=list)
    #depth_I: list = field(default_factory=list)
    #depth_P: list = field(default_factory=list)
    #fwhm: list = field(default_factory=list)
    #bandwidth: list = field(default_factory=list)
    #ell_knee: list = field(default_factory=list)
    #alpha_knee: list = field(default_factory=list)
    #channels_tags: list = field(default_factory=list)
    def __init__(self):
        # Initialize all attributes to None or default values
        self.frequency = None
        self.depth_I = None
        self.depth_P = None
        self.fwhm = None
        self.bandwidth = None
        self.ell_knee = None
        self.alpha_knee = None
        self.channels_tags = []
        self.path_bandpasses = None
        self.path_hits_maps = None
        self.path_depth_maps = None
        self.path_beams = None
        self.beams = None

    def reset_attributes(self):
        """Reset all attributes to their default values."""
        self.__init__()

    def remove_none_attributes(self):
        """Remove attributes that are set to None."""
        for attr in list(self.__dict__.keys()):  # Use list() to avoid runtime modification issues
            if getattr(self, attr) is None:
                delattr(self, attr)

    def load_from_yaml(self, yaml_data: Dict[str, Any], experiment: str):
        # Reset attributes before loading new experiment data
        self.reset_attributes()

        experiment_data = yaml_data.get(experiment, {})

        attrs = [
            'frequency', 'depth_I', 'depth_P', 'fwhm', 'bandwidth',
            'path_bandpasses', 'ell_knee', 'alpha_knee',
            'path_hits_maps', 'path_depth_maps'
        ]

        for attr in attrs:
            if attr in experiment_data:
                setattr(self, attr, experiment_data[attr])

        self.beams = experiment_data.get("beams", "gaussian")
        if self.beams != "gaussian":
            if 'path_beams' not in experiment_data:
                raise ValueError("Missing 'path_beams' for non-gaussian beams in the experiment yaml file.")
            self.path_beams = experiment_data['path_beams']
        else:
            if 'fwhm' not in experiment_data:
                raise ValueError(f"FWHM must be provided in the yaml file for gaussian beams.")

#        for attr in ['path_bandpasses', 'path_hits_maps', 'path_depth_maps', 'path_beams']:
#            if hasattr(self, attr):
#                setattr(self, attr, os.path.join(path_utils, getattr(self, attr)))

        self.channels_tags = experiment_data.get("channels_tags", [])
        if not self.channels_tags:
            self._generate_channel_tags()

        # Remove attributes that are None after loading
        self.remove_none_attributes()


    def _generate_channel_tags(self):
        self.channels_tags = []
        unique_freqs, counts = np.unique(self.frequency, return_counts=True)
        labels = [list(string.ascii_lowercase[:c]) for c in counts]

        for freq in self.frequency:
            idx = np.where(unique_freqs == freq)[0][0]
            label = labels[idx].pop(0)
            tag = f"{freq}GHz" if counts[idx] == 1 else f"{freq}{label}GHz"
            self.channels_tags.append(tag)
        
@dataclass
class Configs:
    """
    Class to store settings and relevant quantities for the main script.
    """

    config_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if self.config_path:
            self.config = yaml.load(open(self.config_path), Loader=yaml.FullLoader)
        self._store_passed_settings()
        if self.experiment:
            self.instrument = InstrumentConfig()
            self._load_experiment_parameters()
            if self.generate_input_foregrounds and self.bandpass_integrate:
                    if not hasattr(self.instrument, 'bandwidth') and not hasattr(self.instrument, 'path_bandpasses'):
                        raise ValueError(f"If bandpass_integrate is True, 'bandwidth' or 'path_bandpasses' must be provided in the experiment yaml file.")
            self.bring_to_common_resolution = self.config.get("bring_to_common_resolution", True)
        

    def _store_passed_settings(self):
        self.nside = self.config["nside"]
        self.data_type = self.config["data_type"]

#        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        base_dir = os.path.abspath(os.path.dirname(__file__))

        self.lmin = self.config.get("lmin") or 2    
        self.lmax = self.config.get("lmax") or 2 * self.nside
        self.nside_in = self.config.get("nside_in") or self.nside
        self.fwhm_out = self.config.get("fwhm_out") or 2.5 * hp.nside2resol(self.nside, arcmin=True)
        self.verbose = self.config.get("verbose", False)
        self.nsim_start = self.config.get("nsim_start") or 0
        self.nsims = self.config.get("nsims") or 1
        self.parallelize = self.config.get("parallelize", False)
#        self.path_utils = self.config.get("path_utils") or os.path.join(base_dir, "utils")
        self.compsep = self.config.get("compsep") or ""
        self.compsep_residuals = self.config.get("compsep_residuals") or ""
        self.real_mc_tracers = self.config.get("real_mc_tracers") or ""
        self.combine_outputs = self.config.get("combine_outputs") or ""
        self.foreground_models = self.config.get("foreground_models") or ["d0","s0"]
        self.field_in = self.config.get("field_in") or "TQU"
        self.field_out = self.config.get("field_out") or "TQU"
        self.experiments_file = self.config.get("experiments_file") or os.path.join(base_dir, "utils", "experiments.yaml") 
        self.experiment = self.config.get("experiment") or ""
        self.pixel_window_in = self.config.get("pixel_window_in", False)
        self.pixel_window_out = self.config.get("pixel_window_out", False)
        self.units = self.config.get("units") or "uK_CMB"
        self.coordinates = self.config.get("coordinates") or "G"
#        self.mask_path = self.config.get("mask_path")
#        self.mask_type = self.config.get("mask_type") or "mask_for_compsep"
        self.mask_observations = self.config.get("mask_observations", None)
        self.mask_covariance = self.config.get("mask_covariance", None)
        self.leakage_correction = self.config.get("leakage_correction", None)
        if self.compsep or self.compsep_residuals or self.combine_outputs:
            self.save_compsep_products = self.config.get("save_compsep_products", True)
            self.return_compsep_products = self.config.get("return_compsep_products", False)
            if not self.save_compsep_products and not self.return_compsep_products:
                raise ValueError("At least one of save_compsep_products and return_compsep_products must be True.")
        self.path_outputs = self.config.get("path_outputs") or os.path.join(os.getcwd(), "outputs", self.experiment, ''.join(self.foreground_models))
#        self.path_outputs = os.path.normpath(os.path.join(os.getcwd(), self.path_outputs))

        self.generate_input_foregrounds = self.config.get("generate_input_foregrounds", True)
        self.generate_input_noise = self.config.get("generate_input_noise", True)
        self.generate_input_cmb = self.config.get("generate_input_cmb", True)
        self.generate_input_data = self.config.get("generate_input_data", True)
        self.bandpass_integrate = self.config.get("bandpass_integrate", False)
        if self.generate_input_foregrounds or self.generate_input_noise or self.generate_input_cmb or self.generate_input_data:
            self.save_inputs = self.config.get("save_inputs", False)
        if self.generate_input_noise:
            self.seed_noise = self.config.get("seed_noise", None)
        if self.generate_input_cmb:
            self.seed_cmb = self.config.get("seed_cmb", None)
        self.cls_cmb_path = self.config.get("cls_cmb_path") or os.path.join(base_dir, "utils", "Cls_Planck2018_lensed_r0.fits")
        self.cls_cmb_new_ordered = self.config.get("cls_cmb_new_ordered", True)

        # Input/output paths   
        dataname = f"total_{self.data_type}_ns{self.nside}_lmax{self.lmax}"
        def_data_path = os.path.join(os.getcwd(), "inputs", self.experiment, "total", ''.join(self.foreground_models), dataname)
        self.data_path = self.config.get("data_path") or def_data_path
#        self.data_path = os.path.normpath(os.path.join(os.getcwd(), self.data_path))

        noisename = f"noise_{self.data_type}_ns{self.nside}_lmax{self.lmax}"
        def_noise_path = os.path.join(os.getcwd(), "inputs", self.experiment, "noise", noisename)
        self.noise_path = self.config.get("noise_path") or def_noise_path
#        self.noise_path = os.path.normpath(os.path.join(os.getcwd(), self.noise_path))

        cmbname = f"cmb_{self.data_type}_ns{self.nside}_lmax{self.lmax}"
        def_cmb_path = os.path.join(os.getcwd(), "inputs", self.experiment, "cmb", cmbname)
        self.cmb_path = self.config.get("cmb_path") or def_cmb_path
#        self.cmb_path = os.path.normpath(os.path.join(os.getcwd(), self.cmb_path))

        fgdsname = f"foregrounds_{self.data_type}_ns{self.nside}_lmax{self.lmax}"
        def_fgds_path = os.path.join(os.getcwd(), "inputs", self.experiment, "foregrounds", ''.join(self.foreground_models), fgdsname)
        self.fgds_path = self.config.get("fgds_path") or def_fgds_path
#        self.fgds_path = os.path.normpath(os.path.join(os.getcwd(), self.fgds_path))

        self.return_fgd_components = self.config.get("return_fgd_components", False)

        self._validate_paths()

        self.compute_spectra = self.config.get("compute_spectra") or ""
        if self.compute_spectra:
            self.delta_ell = self.config.get("delta_ell") or 1
            self.spectra_comp = self.config.get("spectra_comp") or "anafast"
            self.return_Dell = self.config.get("return_Dell", False)
            self.field_cls_out = self.config.get("field_cls_out") or self.field_out
            self.save_spectra = self.config.get("save_spectra", True)
            self.return_spectra = self.config.get("return_spectra", True)
            self.save_mask = self.config.get("save_mask", True)
            
    def _validate_paths(self):
        for name, flag in zip(
            ['fgds_path', 'noise_path', 'cmb_path', 'data_path'],
            [self.generate_input_foregrounds, self.generate_input_noise, self.generate_input_cmb, self.generate_input_data]
        ):
            if ((not flag) or self.save_inputs) and not getattr(self, name):
                raise ValueError(f"Path '{name}' must be specified.")

    def _load_experiment_parameters(self):
        experiments_yaml_path = self.experiments_file
        if os.path.exists(experiments_yaml_path):
            with open(experiments_yaml_path, 'r') as file:
                experiments_data = yaml.safe_load(file)
                self.instrument.load_from_yaml(experiments_data, self.experiment)

    def to_dict_for_mc(self) -> Dict[str, Any]:
        """
        Extract relevant attributes as a dictionary.
        """
        return {
            'lmin': self.lmin,
            'lmax': self.lmax,
            'nside': self.nside,
            'nside_in': self.nside_in,
            'data_type': self.data_type,
            'fwhm_out': self.fwhm_out,
            'foreground_models': self.foreground_models,
            'experiment': self.experiment,
            'pixel_window_in': self.pixel_window_in,
#            'pixel_window_out': self.pixel_window_out,
            'units': self.units,
            'coordinates': self.coordinates,
            'bandpass_integrate': self.bandpass_integrate,
#            'mask_path': self.mask_path,
            'mask_observations': self.mask_observations,
            'mask_covariance': self.mask_covariance,
            'instrument': self.instrument,
            'verbose': self.verbose,
            'cls_cmb_path': self.cls_cmb_path,
            'cls_cmb_new_ordered': self.cls_cmb_new_ordered,
            'experiments_file': self.experiments_file,
        }


__all__ = [
    name
    for name, obj in globals().items()
    if callable(obj) and getattr(obj, "__module__", None) == __name__
] 