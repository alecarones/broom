import os
import sys
from dataclasses import dataclass, field
import yaml
from tqdm import tqdm
from typing import Dict, Any, Optional
import string
import numpy as np

from .utils import (
    join,
    parse_args,
)

def get_params(config_path: Optional[str] = None) -> "Configs":
    yaml.add_constructor("!join", join)

    if config_path is None:
        parsed_args = parse_args()
        config_path = parsed_args.config_path
        config_path = os.path.abspath(config_path)

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

    def load_from_yaml(self, yaml_data: Dict[str, Any], experiment: str):
        experiment_data = yaml_data.get(experiment, {})
        
        if 'frequency' in experiment_data:
            self.frequency = experiment_data['frequency']
        if 'depth_I' in experiment_data:
            self.depth_I = experiment_data['depth_I']
        if 'depth_P' in experiment_data:
            self.depth_P = experiment_data['depth_P']
        if 'fwhm' in experiment_data:
            self.fwhm = experiment_data['fwhm']
        if 'bandwidth' in experiment_data:
            self.bandwidth = experiment_data['bandwidth']
        if 'path_bandpasses' in experiment_data:
            self.path_bandpasses = experiment_data['path_bandpasses']
        if 'ell_knee' in experiment_data:
            self.ell_knee = experiment_data['ell_knee']
        if 'alpha_knee' in experiment_data:
            self.alpha_knee = experiment_data['alpha_knee']
        if 'path_hits_maps' in experiment_data:
            self.path_hits_maps = experiment_data['path_hits_maps']
        if 'path_depth_maps' in experiment_data:
            self.path_depth_maps = experiment_data['path_depth_maps']
        if 'beams' in experiment_data:
            self.beams = experiment_data['beams']
        else:
            self.beams = "gaussian"
        if self.beams != "gaussian":
            if path_beams not in experiment_data:
                raise ValueError(f"Path to the beams must be provided in the yaml file as 'path_beams' for {self.beams} beams.")
            self.path_beams = experiment_data['path_beams']
        else:
            if 'fwhm' not in experiment_data:
                raise ValueError(f"FWHM must be provided in the yaml file as 'fwhm' for gaussian beams.")
        if 'channels_tags' in experiment_data:
            self.channels_tags = experiment_data['channels_tags']
        else:
            self.channels_tags = []
            unique_freqs, counts = np.unique(self.frequency, return_counts=True)
            labels = [list(string.ascii_lowercase[:count]) for count in counts]
            for idx, freq in enumerate(self.frequency):
                if counts[unique_freqs == freq] == 1:
                    self.channels_tags.append(f"{freq}GHz")
                else:
                    self.channels_tags.append(f"{freq}{labels[np.argwhere(unique_freqs == freq)[0][0]][0]}GHz")
                    labels[np.argwhere(unique_freqs == freq)[0][0]].remove(labels[np.argwhere(unique_freqs == freq)[0][0]][0])  # Remove the first element 
                    

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
            if self.generate_input_foregrounds:
                if self.bandpass_integrate:
                    if not hasattr(self.instrument, 'bandwidth') and not hasattr(self.instrument, 'path_bandpasses'):
                        raise ValueError(f"If bandpass_integrate is set to True, either 'bandwidth' or 'path_bandpasses' must be provided in the experiment yaml file.")
            self.bring_to_common_resolution = self.config.get("bring_to_common_resolution", True)
        

    def _store_passed_settings(self):
        self.lmin = self.config.get("lmin", 2)
        self.lmax = self.config["lmax"]
        self.nside = self.config["nside"]
        self.nside_in = self.config.get("nside_in", self.nside)
        self.data_type = self.config["data_type"]
        self.fwhm_out = self.config.get("fwhm_out", 0.)
        self.verbose = self.config.get("verbose", False)
        self.nsim_start = self.config.get("nsim_start", 0)
        self.nsims = self.config.get("nsims", 1)
        self.parallelize = self.config.get("parallelize", False)
        self.compsep = self.config.get("compsep", "")
        self.compsep_residuals = self.config.get("compsep_residuals", "")
        self.real_mc_tracers = self.config.get("real_mc_tracers", "")
        self.foreground_models = self.config.get("foreground_models", ["d0","s0"])
        self.field_in = self.config.get("field_in", "TQU")
        self.field_out = self.config.get("field_out", "")        
        self.experiment = self.config.get("experiment", "")
        self.pixel_window_in = self.config.get("pixel_window_in", False)
        self.pixel_window_out = self.config.get("pixel_window_out", False)
        self.units = self.config.get("units", "uK_CMB")
        self.coordinates = self.config.get("coordinates", "G")
        self.mask_path = self.config.get("mask_path", None)
        self.mask_type = self.config.get("mask_type", "mask_for_compsep")
        self.leakage_correction = self.config.get("leakage_correction", None)
        if self.compsep or self.compsep_residuals:
            self.save_compsep_products = self.config.get("save_compsep_products", True)
            self.return_compsep_products = self.config.get("return_compsep_products", False)
            if not self.save_compsep_products and not self.return_compsep_products:
                raise ValueError("At least one of save_compsep_products and return_compsep_products must be True.")
            if self.save_compsep_products:
                self.path_outputs = self.config.get("path_outputs", os.getcwd() + "/outputs")
#                self.labels_outputs = self.config.get("labels_outputs", "")
#                if not self.labels_outputs:
#                    raise ValueError("Labels for the output files must be provided.")
        self.generate_input_foregrounds = self.config.get("generate_input_foregrounds", True)
        self.generate_input_noise = self.config.get("generate_input_noise", True)
        self.generate_input_cmb = self.config.get("generate_input_cmb", True)
        self.generate_input_data = self.config.get("generate_input_data", True)
        self.bandpass_integrate = self.config.get("bandpass_integrate",False)
        if self.generate_input_foregrounds or self.generate_input_noise or self.generate_input_cmb or self.generate_input_data:
            self.save_inputs = self.config.get("save_inputs", False)
        if self.generate_input_noise:
            self.seed_noise = self.config.get("seed_noise", None)
        if self.generate_input_cmb:
            self.seed_cmb = self.config.get("seed_cmb", None)
            self.cls_cmb_path = self.config.get("cls_cmb_path", "")
        self.data_path = self.config.get("data_path", os.getcwd() + f"/inputs/{self.experiment}/total/")
        self.noise_path = self.config.get("noise_path", os.getcwd() + f"/inputs/{self.experiment}/noise/")
        self.cmb_path = self.config.get("cmb_path", os.getcwd() + f"/inputs/{self.experiment}/cmb/")
        self.fgds_path = self.config.get("fgds_path", os.getcwd() + f"/inputs/{self.experiment}/foregrounds/{''.join(self.foreground_models)}/")
        self.return_fgd_components = self.config.get("return_fgd_components", False)
        
        if (not self.generate_input_foregrounds) or self.save_inputs:
            if not self.fgds_path:
                raise ValueError("The full path to load/store the input foregrounds must be provided as 'fgds_path'.")
        if (not self.generate_input_noise) or self.save_inputs:
            if not self.noise_path:
                raise ValueError("The full path to load/store the input noise maps must be provided as 'noise_path'.")
        if (not self.generate_input_cmb) or self.save_inputs:
            if not self.cmb_path:
                raise ValueError("The full path to load/store the input CMB maps must be provided as 'cmb_path'.")
        if (not self.generate_input_data) or self.save_inputs:
            if not self.data_path:
                raise ValueError("The full path to load/store the input coadded simulations must be provided as 'data_path'.")

        self.compute_spectra = self.config.get("compute_spectra", "")
        if self.compute_spectra:
            self.delta_ell = self.config.get("delta_ell", 1)
            self.spectra_comp = self.config.get("spectra_comp", "anafast")
            self.return_Dell = self.config.get("return_Dell", False)
            self.field_cls_out = self.config.get("field_cls_out", self.field_out)
            self.save_spectra = self.config.get("save_spectra", True)
            self.return_spectra = self.config.get("return_spectra", True)
            

    def _load_experiment_parameters(self):
        experiment = self.config["experiment"]
        experiments_yaml_path = os.path.join("utils", "experiments.yaml")
        if os.path.exists(experiments_yaml_path):
            with open(experiments_yaml_path, 'r') as file:
                experiments_data = yaml.safe_load(file)
                self.instrument.load_from_yaml(experiments_data, experiment)

    def to_dict_for_mc(self) -> Dict[str, Any]:
        """
        Extract relevant attributes as a dictionary.
        """
        return {
            'lmin': self.lmin,
            'lmax': self.lmax,
            'nside': self.nside,
            'data_type': self.data_type,
            'fwhm_out': self.fwhm_out,
            'foreground_models': self.foreground_models,
            'experiment': self.experiment,
            'pixel_window_in': self.pixel_window_in,
            'pixel_window_out': self.pixel_window_out,
            'units': self.units,
            'coordinates': self.coordinates,
            'bandpass_integrate': self.bandpass_integrate,
            'mask_path': self.mask_path,
            'verbose': self.verbose}

  
    