<img src="logo_broom.jpg" width="200"> 

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

- [numpy>1.18.5](https://numpy.org/)
- [scipy>=1.8](https://scipy.org/)
- [healpy>=1.15](https://healpy.readthedocs.io/)
- [pysm3>=3.3.2](https://pysm3.readthedocs.io/en/latest/#)
- [mtneedlet>=0.0.5](https://javicarron.github.io/mtneedlet/)
- [sciki-learn>=1.7.0](https://scikit-learn.org/stable/)


