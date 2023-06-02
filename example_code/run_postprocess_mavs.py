import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import pandas as pd 
import xarray as xr
import sys
# sys.path.append("/home/simon/Documents/GitHub/funpy")
# import funpy.postprocess as fp 
sys.path.append("/home/simon/Documents/GitHub/")
import funpy.postprocess as fp 

rundir = '/media/simon/Seagate Backup Plus Drive/metocean/R&D/SWASH_BigWaveModelling/FUNWAVE/mavs/'
fdir = os.path.join(rundir,'output270')
fdir = os.path.join(rundir,'output290')
# fdir = os.path.join(rundir,'output300')

savedir = os.path.join(rundir, 'postprocessing', 'compiled_output')
if not os.path.exists(savedir):os.makedirs(savedir)

# if not os.path.exists(os.path.join(fdir, 'compiled_output', 'dep.out')):
# 	os.system('cp /data2/enuss/funwave_lab_setup/'+rundir+'/output/dep.out /data2/enuss/funwave_lab_setup/'+rundir+'/postprocessing/compiled_output/')

dx = 2
dy = 2
dt = 1.0

fp.output2netcdf(fdir, savedir, dx, dy, dt, ['eta','u','v','mask','nubrk'])

if False :
    # fp.output2netcdf(fdir, savedir, dx, dy, dt, 'u')
    # fp.output2netcdf(fdir, savedir, dx, dy, dt, 'v')
    # fp.output2netcdf(fdir, savedir, dx, dy, dt, 'mask')
    # fp.output2netcdf(fdir, savedir, dx, dy, dt, 'nubrk')

    fp.uv2vorticity(savedir)

    uu = xr.open_dataset(os.path.join(savedir, 'u.nc'))['u']
    vv = xr.open_dataset(os.path.join(savedir, 'v.nc'))['v']

    fp.compute_fbr(uu, 'fbrx', fdir)
    fp.compute_fbr(vv, 'fbry', fdir)

    fp.crest_identification(fdir)
