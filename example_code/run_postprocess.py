import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import pandas as pd 
import xarray as xr
import funpy.postprocess as fp 

rundir = 'mono_test_ata'
fdir = os.path.join('/data2','enuss','funwave_lab_setup',rundir,'output')
savedir = os.path.join('/data2', 'enuss', 'funwave_lab_setup', rundir, 'postprocessing', 'compiled_output')
if not os.path.exists(savedir):
	os.makedirs(savedir)

if not os.path.exists(os.path.join(fdir, 'compiled_output', 'dep.out')):
	os.system('cp /data2/enuss/funwave_lab_setup/'+rundir+'/output/dep.out /data2/enuss/funwave_lab_setup/'+rundir+'/postprocessing/compiled_output/')

dx = 0.05
dy = 0.1 
dt = 0.2 

fp.output2netcdf(fdir, savedir, dx, dy, dt, 'eta')
fp.output2netcdf(fdir, savedir, dx, dy, dt, 'u')
fp.output2netcdf(fdir, savedir, dx, dy, dt, 'v')
fp.output2netcdf(fdir, savedir, dx, dy, dt, 'mask')
fp.output2netcdf(fdir, savedir, dx, dy, dt, 'nubrk')

fp.uv2vorticity(savedir)

u = xr.open_dataset(os.path.join(fdir, 'u.nc'))['u']
v = xr.open_dataset(os.path.join(fdir, 'v.nc'))['v']

fp.compute_fbr(u, 'fbrx', fdir)
fp.compute_fbr(v, 'fbry', fdir)

fp.crest_identification(fdir)
