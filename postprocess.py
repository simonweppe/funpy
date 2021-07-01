import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import pandas as pd 
import xarray as xr

def funwave_to_netcdf(flist, x, y, time, fpath, name):
	""" Function that takes list of FUNWAVE-TVD text file output and 
	creates a xarray data array and saves to a netcdf file
	"""
	var = np.zeros((len(time),len(y),len(x)))
	for i in range(len(time)):
		var_i = pd.read_csv(os.path.join(fdir,flist[i]), header=None, delim_whitespace=True)
		var_i = np.asarray(var_i)
		var[i,:,:] = var_i
		del var_i
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
			funwave_to_netcdf(flist[s], x, y, time[s], fpath, varname)
		if (i+1)*N < fnum - 1:
			s = slice((i+1)*N, fnum)
			fpath = os.path.join(savedir, '%s_%d.nc' % (varname, i+1))
			funwave_to_netcdf(flist[s], x, y, time[s], fpath, varname)			
	else:
		fpath = os.path.join(savedir, '%s.nc' % varname)
		funwave_to_netcdf(flist, x, y, time, fpath, varname)

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




