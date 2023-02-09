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

plt.ion()
plt.style.use('ggplot')

def run_vel_decomposition(fdir, dx, dy, dt, ufile, vfile, maskfile, upsifile, vpsifile, uphifile, vphifile):
	print("vel decomposition function begins")
	u, x, y = mod_utils.load_masked_variable(fdir, 'u', ufile, 'mask', maskfile)
	v, x, y = mod_utils.load_masked_variable(fdir, 'v', vfile, 'mask', maskfile)
	print("loaded u, v")

	psi, u_psi, v_psi, phi, u_phi, v_phi = mod_utils.vel_decomposition(u, v, dx, dy)
	print("finished vel decomposition")

	fp.var_to_netcdf(u_psi, x, y, dt, 'u_psi', os.path.join(fdir, upsifile))
	fp.var_to_netcdf(v_psi, x, y, dt, 'v_psi', os.path.join(fdir, vpsifile))
	fp.var_to_netcdf(u_phi, x, y, dt, 'u_phi', os.path.join(fdir, uphifile))
	fp.var_to_netcdf(v_phi, x, y, dt, 'v_phi', os.path.join(fdir, vphifile))
	print("finished saving netcdf files")

dx = 0.05; dy = 0.1; dt = 0.2


rundir = 'hmo25_dir1'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)


run_vel_decomposition(fdir, dx, dy, dt, 'u_0.nc', 'v_0.nc', 'mask_0.nc', 'u_psi_0.nc', 'v_psi_0.nc', 'u_phi_0.nc', 'v_phi_0.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_1.nc', 'v_1.nc', 'mask_1.nc', 'u_psi_1.nc', 'v_psi_1.nc', 'u_phi_1.nc', 'v_phi_1.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_2.nc', 'v_2.nc', 'mask_2.nc', 'u_psi_2.nc', 'v_psi_2.nc', 'u_phi_2.nc', 'v_phi_2.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_3.nc', 'v_3.nc', 'mask_3.nc', 'u_psi_3.nc', 'v_psi_3.nc', 'u_phi_3.nc', 'v_phi_3.nc')

rundir = 'hmo25_dir5_cfl09'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

run_vel_decomposition(fdir, dx, dy, dt, 'u_0.nc', 'v_0.nc', 'mask_0.nc', 'u_psi_0.nc', 'v_psi_0.nc', 'u_phi_0.nc', 'v_phi_0.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_1.nc', 'v_1.nc', 'mask_1.nc', 'u_psi_1.nc', 'v_psi_1.nc', 'u_phi_1.nc', 'v_phi_1.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_2.nc', 'v_2.nc', 'mask_2.nc', 'u_psi_2.nc', 'v_psi_2.nc', 'u_phi_2.nc', 'v_phi_2.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_3.nc', 'v_3.nc', 'mask_3.nc', 'u_psi_3.nc', 'v_psi_3.nc', 'u_phi_3.nc', 'v_phi_3.nc')

rundir = 'hmo25_dir10'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

run_vel_decomposition(fdir, dx, dy, dt, 'u_0.nc', 'v_0.nc', 'mask_0.nc', 'u_psi_0.nc', 'v_psi_0.nc', 'u_phi_0.nc', 'v_phi_0.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_1.nc', 'v_1.nc', 'mask_1.nc', 'u_psi_1.nc', 'v_psi_1.nc', 'u_phi_1.nc', 'v_phi_1.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_2.nc', 'v_2.nc', 'mask_2.nc', 'u_psi_2.nc', 'v_psi_2.nc', 'u_phi_2.nc', 'v_phi_2.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_3.nc', 'v_3.nc', 'mask_3.nc', 'u_psi_3.nc', 'v_psi_3.nc', 'u_phi_3.nc', 'v_phi_3.nc')

rundir = 'hmo25_dir20'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

run_vel_decomposition(fdir, dx, dy, dt, 'u_0.nc', 'v_0.nc', 'mask_0.nc', 'u_psi_0.nc', 'v_psi_0.nc', 'u_phi_0.nc', 'v_phi_0.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_1.nc', 'v_1.nc', 'mask_1.nc', 'u_psi_1.nc', 'v_psi_1.nc', 'u_phi_1.nc', 'v_phi_1.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_2.nc', 'v_2.nc', 'mask_2.nc', 'u_psi_2.nc', 'v_psi_2.nc', 'u_phi_2.nc', 'v_phi_2.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_3.nc', 'v_3.nc', 'mask_3.nc', 'u_psi_3.nc', 'v_psi_3.nc', 'u_phi_3.nc', 'v_phi_3.nc')

rundir = 'hmo25_dir30'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

run_vel_decomposition(fdir, dx, dy, dt, 'u_0.nc', 'v_0.nc', 'mask_0.nc', 'u_psi_0.nc', 'v_psi_0.nc', 'u_phi_0.nc', 'v_phi_0.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_1.nc', 'v_1.nc', 'mask_1.nc', 'u_psi_1.nc', 'v_psi_1.nc', 'u_phi_1.nc', 'v_phi_1.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_2.nc', 'v_2.nc', 'mask_2.nc', 'u_psi_2.nc', 'v_psi_2.nc', 'u_phi_2.nc', 'v_phi_2.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_3.nc', 'v_3.nc', 'mask_3.nc', 'u_psi_3.nc', 'v_psi_3.nc', 'u_phi_3.nc', 'v_phi_3.nc')

rundir = 'hmo25_dir40'
rootdir = os.path.join('/data2','enuss','lab_bathy_test','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

run_vel_decomposition(fdir, dx, dy, dt, 'u_0.nc', 'v_0.nc', 'mask_0.nc', 'u_psi_0.nc', 'v_psi_0.nc', 'u_phi_0.nc', 'v_phi_0.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_1.nc', 'v_1.nc', 'mask_1.nc', 'u_psi_1.nc', 'v_psi_1.nc', 'u_phi_1.nc', 'v_phi_1.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_2.nc', 'v_2.nc', 'mask_2.nc', 'u_psi_2.nc', 'v_psi_2.nc', 'u_phi_2.nc', 'v_phi_2.nc')
run_vel_decomposition(fdir, dx, dy, dt, 'u_3.nc', 'v_3.nc', 'mask_3.nc', 'u_psi_3.nc', 'v_psi_3.nc', 'u_phi_3.nc', 'v_phi_3.nc')




















