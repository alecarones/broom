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
    component_separation, estimate_residuals, _combine_products, _load_outputs
)
from .simulations import _get_data_foregrounds_, _get_data_simulations_, _get_noise_simulation, _get_cmb_simulation, _get_full_simulations
from .configurations import Configs, get_params
from .spectra import _compute_spectra, _compute_spectra_, _load_cls
from .clusters import get_and_save_real_tracers_B


__version__ = "0.1.1"
