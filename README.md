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
