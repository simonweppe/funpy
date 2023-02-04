import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import pandas as pd 
import xarray as xr
import funpy.model_utils as mod_utils

def funwave_to_netcdf(fdir, flist, x, y, time, fpath, name):
	""" Function that takes list of FUNWAVE-TVD text file output and 
	creates a xarray data array and saves to a netcdf file
	"""
	var = np.zeros((len(time),len(y),len(x)))
	for i in range(len(time)):
		var_i = pd.read_csv(os.path.join(fdir,flist[i]), header=None, delim_whitespace=True)
		var_i = np.asarray(var_i)
		var[i,:,:] = var_i
		del var_i
	var_to_netcdf(var, x, y, time[1]-time[0], name, fpath)

def var_to_netcdf(var, x, y, dt, name, fpath):
	time = np.linspace(0, len(var)*dt, len(var))
	dim = ["time", "y", "x"]
	coords = [time, y, x]
	dat = xr.DataArray(var, coords=coords, dims=dim, name=name)
	dat.to_netcdf(fpath)

def output2netcdf(fdir, savedir, dx, dy, dt, varname, nchunks=1):
	""" Compiles list of FUNWAVE-TVD text file output for a given variable,
	and uses funwave_to_netcdf to save output into a netcdf file. For file 
	size concerns, output can be saved into N netcdf files, set by nchunks.
	"""
	## load depth file and get x, y dimensions 
	depFile = os.path.join(fdir,'dep.out')
	dep = np.loadtxt(depFile)
	[n,m] = dep.shape

	# x and y field vectors 
	x = np.arange(0,m*dx,dx)
	y = np.arange(0,n*dy,dy)

	# output file list
	flist = [file for file in glob.glob(os.path.join(fdir,'%s_*' % varname))]
	fnum = len(flist)
	time = np.arange(0,fnum)*dt 
	if nchunks>1:
		for i in range(nchunks):
			N = int(len(time)/nchunks)
			s = slice(i*N, (i+1)*N)
			fpath = os.path.join(savedir, '%s_%d.nc' % (varname, i))
			funwave_to_netcdf(fdir, flist[s], x, y, time[s], fpath, varname)
		if (i+1)*N < fnum - 1:
			s = slice((i+1)*N, fnum)
			fpath = os.path.join(savedir, '%s_%d.nc' % (varname, i+1))
			funwave_to_netcdf(fdir, flist[s], x, y, time[s], fpath, varname)			
	else:
		fpath = os.path.join(savedir, '%s.nc' % varname)
		funwave_to_netcdf(fdir, flist, x, y, time, fpath, varname)

def uv2vorticity(fdir, savefile = 'vorticity.nc', savemask = 'vorticity_mask.nc', ufile = 'u.nc', vfile = 'v.nc', maskfile = 'mask.nc'):
	""" Takes compiled (netcdf) u and v velocity output from FUNWAVE-TVD and computes du/dy,
	dv/dx, averages each to get them onto the same grid, and then computes the vorticity 
	(dv/dx - du/dy) and saves to a netcdf file. The vorticity mask will also be saved.
	"""
	u_dat = xr.open_dataset(os.path.join(fdir, ufile))
	# load time, x, y and compute dx, dy
	time = u_dat.time.values
	x = np.asarray(u_dat.x)
	y = np.asarray(u_dat.y)
	dx = x[1]-x[0]
	dy = y[1]-y[0]

	u = u_dat['u'].values
	dudy = np.asarray([(u[i,1:,:]-u[i,:-1,:])/dy for i in range(len(u))])
	del u, u_dat

	v_dat = xr.open_dataset(os.path.join(fdir, vfile))
	v = v_dat['v'].values
	dvdx = np.asarray([(v[i,:,1:]-v[i,:,:-1])/dx for i in range(len(v))])
	del v, v_dat

	# average in y to get dvdx on the same grid 
	dvdx_avg = np.asarray([(dvdx[i,1:,:]+dvdx[i,:-1,:])/2 for i in range(len(dvdx))])

	# average in x to get dudy on the same grid
	dudy_avg = np.asarray([(dudy[i,:,1:]+dudy[i,:,:-1])/2 for i in range(len(dudy))])

	vor = dvdx_avg - dudy_avg
	xbar = (x[1:]+x[:-1])/2
	ybar = (y[1:]+y[:-1])/2
	[xx,yy] = np.meshgrid(xbar, ybar)

	dim = ["time", "y", "x"]
	coords = [time, ybar, xbar]
	dat = xr.DataArray(vor, coords=coords, dims=dim, name='vorticity')
	dat.to_netcdf(os.path.join(fdir, savefile))

	## vorticity mask
	mask = xr.open_dataset(os.path.join(fdir, maskfile))
	mask = mask['mask'].values
	vorticity_mask = np.asarray([mask[i,:-1,:-1]*mask[i,:-1,1:]*mask[i,1:,:-1]*mask[i,1:,1:] for i in range(len(mask))])
	mask_dat = xr.DataArray(vorticity_mask, coords=coords, dims=dim, name='vorticity_mask')
	mask_dat.to_netcdf(os.path.join(fdir, savemask))

def compute_fbr(vel, name, fdir, savefile='fbr.nc', nufile='nubrk.nc', etafile='eta.nc', depfile='dep.out', dx=0.05, dy=0.1, dt=0.2):
	nubrk_dat = xr.open_dataset(os.path.join(fdir, nufile))
	eta_dat = xr.open_dataset(os.path.join(fdir, etafile))
	dep = np.loadtxt(os.path.join(fdir, depfile))
	nubrk = nubrk_dat['nubrk']
	eta = eta_dat['eta']
	x = eta_dat['x']
	y = eta_dat['y']

	dudx = np.gradient(vel, dx, axis=2)
	dudy = np.gradient(vel, dy, axis=1)
	heta = np.asarray([dep + eta[i,:,:] for i in range(len(eta))])

	term1 = nubrk * heta * dudx 
	term2 = nubrk * heta * dudy

	del vel, eta, nubrk, dep 

	term1dx = np.gradient(term1, dx, axis=2)
	term2dy = np.gradient(term2, dy, axis=1)

	fbr = 1/heta * (term1dx + term2dy)	

	T = len(fbr)
	dim = ["time", "y", "x"]
	coords = [np.linspace(0,T*dt,T), y, x]
	dat = xr.DataArray(fbr, coords=coords, dims=dim, name=name)
	dat.to_netcdf(os.path.join(fdir, savefile))

def crest_identification(fdir, nufile='nubrk.nc', maskfile='mask.nc', savefile='crest.nc', threshold=0, dt=0.2):
	x, y, nubrk = mod_utils.load_var_lab(fdir, 'nubrk', nufile, 'mask', maskfile)

	T = len(nubrk)
	for t in range(T):
		nubrk_bar, nubin, num_labels, labels = mod_utils.find_crests(nubrk[t,:,:], x, y, threshold=threshold)
		if t == 0:
			labels_total = np.expand_dims(labels, axis = 0)
		else:
			labels_total = np.concatenate((labels_total, np.expand_dims(labels, axis=0)), axis=0)

	dim = ["time", "y", "x"]
	coords = [np.linspace(0,T*dt,T), y, x]
	dat = xr.DataArray(labels_total, coords=coords, dims=dim, name='labels')
	dat.to_netcdf(os.path.join(fdir, savefile))


