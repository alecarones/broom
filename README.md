<img src="logo_broom.png" width="200"> 

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


## 📦 Dependencies

This package relies on several scientific Python libraries:

- [astropy>=6.0.1](https://www.astropy.org/)
- [numpy>1.18.5](https://numpy.org/)
- [scipy>=1.8](https://scipy.org/)
- [healpy>=1.15](https://healpy.readthedocs.io/)
- [pysm3>=3.3.2](https://pysm3.readthedocs.io/en/latest/#)
- [mtneedlet>=0.0.5](https://javicarron.github.io/mtneedlet/)


