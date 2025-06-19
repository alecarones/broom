# BROOM: Blind Reconstruction Of signal from Observations in the Microwaves

**BROOM** is a Python package for blind component separation in Cosmic Microwave Background (CMB) data analysis.

---

## 📦 Installation

You can install the base package using:

```
pip install cmbroom
```

In this case, if you want to use the few functionalities related to pymaster you should have the package already installed with version >= 2.4.

Alternatively, it can be installed automatically with

```
pip install cmbroom[pymaster]
```

but you **must first install the following system libraries** on your machine:

**On Ubuntu/Debian:**
```
sudo apt update
sudo apt install build-essential python3-dev libfftw3-dev libcfitsio-dev
```

**On macOS (with Homebrew):**
```
brew install fftw cfitsio
```
