import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import numpy.ma as ma

def load_masked_variable(fdir, var, varfile, mask, maskfile):
	mask_ds = xr.open_mfdataset(os.path.join(fdir, maskfile))
	var_ds = xr.open_mfdataset(os.path.join(fdir, varfile))
	x = np.asarray(var_ds.x)
	y = np.asarray(var_ds.y)
	var_masked = ma.masked_where(mask_ds[mask].values==0, var_ds[var].values)
	return var_masked, x, y 

def compute_Hsig(eta, start, end):
	return 4*np.nanstd(eta[start:end, :, :], axis=0)

