import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import numpy.ma as ma
import pandas as pd 
import funpy.model_utils as mod_utils
import funpy.postprocess as fp 
import cmocean.cm as cmo 
import re
import glob
from scipy.signal import welch, hanning
from funpy import filter_functions as ff
import datetime

dx = 0.05; dy = 0.1; dt = 0.2

def rms_calc(fdir):
	u_psi_dat = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')
	u_phi_dat = xr.open_mfdataset(os.path.join(fdir, 'u_phi_*.nc'), combine='nested', concat_dim='time')
	u_dat = xr.open_mfdataset([os.path.join(fdir, 'u_0.nc'), os.path.join(fdir, 'u_1.nc'), os.path.join(fdir, 'u_2.nc'), os.path.join(fdir, 'u_3.nc')], combine='nested', concat_dim='time')

	x = u_dat['x']
	y = u_dat['y']

	u_rec = u_psi_dat['u_psi'] + u_phi_dat['u_phi']
	u = u_dat['u']

	xend = int(54/dx)

	f.open(os.path.join(fdir, 'rms_u.txt'))
	N = len(x[:xend])*len(y)

	for t in range(len(u)):
		rms = np.sqrt(np.nansum((u[t,:,:xend]-u_rec[t,:,:xend])**2)/N)
		f.write('%f' % rms)
		f.write('\n')
	f.close()

	del u_psi_dat, u_phi_dat, u_dat, u_rec, u
	v_psi_dat = xr.open_mfdataset(os.path.join(fdir, 'v_psi_*.nc'), combine='nested', concat_dim='time')
	v_phi_dat = xr.open_mfdataset(os.path.join(fdir, 'v_phi_*.nc'), combine='nested', concat_dim='time')
	v_dat = xr.open_mfdataset([os.path.join(fdir, 'v_0.nc'), os.path.join(fdir, 'v_1.nc'), os.path.join(fdir, 'v_2.nc'), os.path.join(fdir, 'v_3.nc')], combine='nested', concat_dim='time')

	x = u_dat['x']
	y = u_dat['y']

	v_rec = v_psi_dat['v_psi'] + u_phi_dat['v_phi']
	v = v_dat['v']

	f.open(os.path.join(fdir, 'rms_v.txt'))
	N = len(x[:xend])*len(y)

	for t in range(len(v)):
		rms = np.sqrt(np.nansum((v[t,:,:xend]-v_rec[t,:,:xend])**2)/N)
		f.write('%f' % rms)
		f.write('\n')
	f.close()


rundir = 'hmo25_dir1'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

rms_calc(fdir)

rundir = 'hmo25_dir5_cfl09'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

rms_calc(fdir)

rundir = 'hmo25_dir10'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

rms_calc(fdir)

rundir = 'hmo25_dir20'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

rms_calc(fdir)

rundir = 'hmo25_dir30'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

rms_calc(fdir)

rundir = 'hmo25_dir40'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

rms_calc(fdir)







