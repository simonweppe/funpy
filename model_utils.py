import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import numpy.ma as ma
from scipy.signal import welch, hanning
from funpy import filter_functions as ff 
import cv2

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

def compute_spec(var, dt, WL=512, OL=256, n=1, axis=0):
	freq, spec = welch(var, fs=1/dt, window='hann', nperseg=WL*n, noverlap=OL*n, axis=axis)
	return freq, spec

def compute_Hsig_spectrally(freq, spec, fmin, fmax):
	ind = np.where((freq>fmin)&(freq<fmax))[0]
	Hs = 4*np.sqrt(np.nansum(spec[ind], axis=0)*np.diff(freq)[0])
	return Hs

def load_var_lab(rootdir, var, filepath, maskvar, maskfile):
	var, x, y = load_masked_variable(os.path.join(rootdir, 'compiled_output'), var, filepath, maskvar, maskfile)
	x = model2lab(x)
	x_ind = np.where(x>0)[0]
	return x[x_ind], y, var[:,:,x_ind]

def binize_var(var, threshold):
	var[var>threshold] = 1
	var[var<=threshold] = 0
	return var 

def spatially_avg(var, x, y, order=1, filtx=0.5, filty=0.5):
	window = ff.lanczos_2Dwindow(y, x, 1, 0.5, 0.5)
	var_bar = ff.lanczos_2D(var.data, var.mask, window, len(y), len(x))
	return var_bar

def find_crests(var, x, y, threshold=0, connectivity=8, order=1, filtx=0.5, filty=0.5):
	var_bar = spatially_avg(var, x, y, order=order, filtx=filtx, filty=filty)
	var_bin = binize_var(var_bar, threshold).astype(np.uint8)
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(var_bin, connectivity=connectivity)

	[xx, yy] = np.meshgrid(x, y)

	crestend_max_x = np.zeros(num_labels)
	crestend_min_x = np.zeros(num_labels)
	crestend_max_y = np.zeros(num_labels)
	crestend_min_y = np.zeros(num_labels)

	for i in range(1,num_labels):
		ind_x = np.where(labels==i)[0]
		ind_y = np.where(labels==i)[1]
		crest_x = xx[ind_x, ind_y]
		crest_y = yy[ind_x, ind_y]
		crestend_max_y[i] = np.max(crest_y)
		crestend_min_y[i] = np.min(crest_y)
		crestend_max_x[i] = crest_x[np.argmax(crest_y)]
		crestend_min_x[i] = crest_x[np.argmin(crest_y)]

	crestlen = crestend_max_y - crestend_min_y
	return var_bar, var_bin, num_labels, labels, crestend_min_x, crestend_max_x, crestend_min_y, crestend_max_y, crestlen
