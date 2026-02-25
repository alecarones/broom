"""

# BROOM: Blind Reconstruction Of signals from Observations in the Microwaves

**BROOM** is a Python package for blind component separation and Cosmic Microwave Background (CMB) data analysis.

---

## 📦 Installation

You can install the base package using:

```
pip install cmbroom
```

This installs the core functionality.  
If you plan to use the few functions that depend on `pymaster`, **you must install it separately** (version `>=2.4`).

---

### 🔧 To include `pymaster` automatically:

You can install `cmbroom` along with its optional `pymaster` dependency by running:

```
pip install cmbroom[pymaster]
```

However, `pymaster` requires some additional system libraries to be installed **before** running the above command.

#### ✅ On Ubuntu/Debian:
```
sudo apt update
sudo apt install build-essential python3-dev libfftw3-dev libcfitsio-dev
```

#### ✅ On macOS (using Homebrew):
```
brew install fftw cfitsio
```

## Documentation

A detailed introduction to the parameters and simulation pipeline is available in:

- [**tutorials/tutorial_satellite.ipynb**](tutorials/tutorial_satellite.ipynb)   
- [**configs/config_demo.yaml**](broom/configs/config_demo.yaml) — Example configuration file

Component separation methods are covered in:

- [**tutorials/tutorial_satellite.ipynb**](tutorials/tutorial_satellite.ipynb) 
- [**tutorials/tutorial_satellite_part2.ipynb**](tutorials/tutorial_satellite_part2.ipynb) 

Power spectrum estimation is demonstrated in:

- [**tutorials/tutorial_spectra.ipynb**](tutorials/tutorial_spectra.ipynb)

For partial-sky, ground-based experiment analysis, see:

- [**tutorials/tutorial_groundbased.ipynb**](tutorials/tutorial_groundbased.ipynb) 

🔗 **Full online documentation:**  
👉 [https://alecarones.github.io/broom/](https://alecarones.github.io/broom/)


## References

Paper on **broom** package is in preparation.

If you use the following methodologies please cite the corresponding papers:

- ILC or NILC: [Bennett et al., 2003](https://arxiv.org/abs/astro-ph/0302207), [Delabrouille et al., 2009](https://arxiv.org/abs/0807.0773)
- cMILC: [Remazeilles et al., 2021](https://arxiv.org/abs/2006.08628), [Carones et al., 2024](https://arxiv.org/abs/2402.17579)
- MC-ILC or MC-NILC: [Carones et al., 2023](https://arxiv.org/abs/2212.04456)
- PILC: [Fernández-Cobos et al., 2016](https://arxiv.org/abs/1601.01515)
- cPILC: [Adak, 2021](https://arxiv.org/abs/2104.13778)
- GILC, GNILC, GPILC: [Remazeilles et al., 2011](https://arxiv.org/abs/1103.1166), [Planck Collaboration, 2016](https://arxiv.org/abs/1605.09387)
- foreground diagnostic: [Carones et al., 2024](https://arxiv.org/abs/2402.17579)
- power spectrum computation: [Gorski et al., 2005](https://arxiv.org/abs/astro-ph/0409513), [Zonca et al., 2019](https://ui.adsabs.harvard.edu/abs/2019JOSS....4.1298Z/abstract), [Alonso et al., 2019](https://arxiv.org/abs/1809.09603)

### 📦 Dependencies

This package relies on several scientific Python libraries:

- [astropy>=6.0.1](https://www.astropy.org/)
- [numpy>1.18.5](https://numpy.org/)
- [scipy>=1.8](https://scipy.org/)
- [healpy>=1.15](https://healpy.readthedocs.io/)
- [pysm3>=3.3.2](https://pysm3.readthedocs.io/en/latest/#)
- [mtneedlet>=0.0.5](https://javicarron.github.io/mtneedlet/)



"""


from .compsep import (
    component_separation, estimate_residuals, _combine_products, _load_outputs, combine_with_weights, get_nuisance_covariance
)
from .simulations import _get_foregrounds_, _get_noise_, _get_cmb_, get_input_data
from .configurations import Configs, get_params
from .spectra import _compute_spectra, _compute_spectra_, _load_cls
from .clusters import get_mcilc_tracers


__version__ = "0.1.5"
