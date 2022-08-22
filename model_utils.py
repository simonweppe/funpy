import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import numpy.ma as ma
from scipy.signal import welch, hanning


def load_masked_variable(fdir, var, varfile, mask, maskfile):
	mask_ds = xr.open_mfdataset(os.path.join(fdir, maskfile))
	var_ds = xr.open_mfdataset(os.path.join(fdir, varfile))
	x = np.asarray(var_ds.x)
	y = np.asarray(var_ds.y)
	var_masked = ma.masked_where(mask_ds[mask].values==0, var_ds[var].values)
	return var_masked, x, y 

def model2lab(x):
	return x - 22

def compute_Hsig(eta, start, end):
	return 4*np.nanstd(eta[start:end, :, :], axis=0)

def compute_SSE_spec(eta, dt, WL=512, OL=256, n=1, axis=0):
	freq, spec = welch(eta, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=axis)
	return freq, spec

def compute_Hsig_spectrally(freq, spec, fmin, fmax):
	ind = np.where((freq>fmin)&(freq<fmax))[0]
	Hs = 4*np.sqrt(np.sum(spec[ind], axis=0)*np.diff(freq)[0])
	return Hs
