import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import numpy.ma as ma

def load_masked_variable(fdir, var, varfile, mask, maskfile):
	mask_ds = xr.open_dataset(os.path.join(fdir, maskfile))
	mask = mask_ds[mask].values
	var_ds = xr.open_dataset(os.path.join(fdir, varfile))
	var = var_ds[var].values
	x = np.asarray(var_ds.x)
	y = np.asarray(var_ds.y)
	var_masked = ma.masked_where(mask==0, var)
	return var, x, y 

def compute_Hsig(eta, start, end):
	return 4*np.nanstd(eta[start:end, :, :], axis=0)

