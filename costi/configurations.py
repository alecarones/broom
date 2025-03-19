import os
import sys
from dataclasses import dataclass, field
import yaml
from tqdm import tqdm
from typing import Dict, Any, Optional

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
    frequency: list = field(default_factory=list)
    depth_I: list = field(default_factory=list)
    depth_P: list = field(default_factory=list)
    fwhm: list = field(default_factory=list)
    bandwidth: list = field(default_factory=list)

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
            if self.generate_input_simulations:
                if self.bandpass_integrate:
                    if not self.instrument.bandwidth:
                        raise ValueError(f"bandpass_integrate is set to True, but bandwidth is not present for {self.experiment}. \
                        Please set bandpass_integrate to False or provide bandwidth.")
            self.bring_to_common_resolution = self.config.get("bring_to_common_resolution", True)
        

    def _store_passed_settings(self):
        self.lmin = self.config["lmin"]
        self.lmax = self.config["lmax"]
        self.nside = self.config ["nside"]
        self.data_type = self.config["data_type"]
        self.fwhm_out = self.config.get("fwhm_out", 0.)
        self.input_beams = self.config.get("input_beams", "guassian")
        self.verbose = self.config.get("verbose", False)
        self.nsim_start = self.config.get("nsim_start", 0)
        self.nsims = self.config.get("nsims", 1)
        self.parallelize = self.config.get("parallelize", False)
        self.compsep_runs = self.config.get("compsep_runs", "")
        self.foreground_models = self.config.get("foreground_models", ["d0","s0"])
        self.field_in = self.config["field_in"]
        self.field_out = self.config.get("field_out", "")        
        self.experiment = self.config.get("experiment", "")
        self.pixel_window_in = self.config.get("pixel_window_in", False)
        self.pixel_window_out = self.config.get("pixel_window_out", False)
        self.units = self.config.get("units", "uK_CMB")
        self.leakage_correction = self.config.get("leakage_correction", None)
        if self.compsep_runs:
            self.save_compsep_products = self.config.get("save_compsep_products", False)
            self.return_compsep_products = self.config.get("return_compsep_products", True)
            if not self.save_compsep_products and not self.return_compsep_products:
                raise ValueError("At least one of save_compsep_products and return_compsep_products must be True.")
            if self.save_compsep_products:
                self.path_outputs = self.config.get("path_outputs", os.getcwd() + "/outputs")
                self.labels_outputs = self.config.get("labels_outputs", "")
                if not self.labels_outputs:
                    raise ValueError("Labels for the output files must be provided.")
        self.return_fgd_components = self.config.get("return_fgd_components", False)
        self.generate_input_simulations = self.config.get("generate_input_simulations", True)
        if self.generate_input_simulations:
            self.save_input_simulations = self.config.get("save_input_simulations", False)
            self.bandpass_integrate = self.config.get("bandpass_integrate",False)
            self.seed_noise = self.config.get("seed_noise", None)
            self.seed_cmb = self.config.get("seed_cmb", None)
            self.ell_knee = self.config.get("ell_knee", None)
            self.alpha_knee = self.config.get("alpha_knee", None)
            self.cls_cmb_path = self.config.get("cls_cmb_path", "")
            if self.save_input_simulations:
                self.inputs_path = self.config.get("inputs_path", "../inputs")
        if not self.generate_input_simulations:
            self.load_input_simulations = self.config.get("load_input_simulations", True)
            if self.load_input_simulations:
                self.data_path = self.config.get("data_path", "")
                self.noise_path = self.config.get("noise_path", "")
                self.cmb_path = self.config.get("cmb_path", "")
                self.fgds_path = self.config.get("fgds_path", "")
                if not self.noise_path or not self.cmb_path or not self.fgds_path:
                    raise ValueError("The paths to the input CMB, noise and foregrounds must be provided.")    
            else:
                print("Warning: No input simulations generated or loaded. You must pass your own inputs to compsep.")
        else:
            self.load_input_simulations = self.config.get("load_input_simulations", False)


    def _load_experiment_parameters(self):
        experiment = self.config["experiment"]
        experiments_yaml_path = os.path.join("data", "experiments.yaml")
        if os.path.exists(experiments_yaml_path):
            with open(experiments_yaml_path, 'r') as file:
                experiments_data = yaml.safe_load(file)
                self.instrument.load_from_yaml(experiments_data, experiment)

  
    