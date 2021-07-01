import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import numpy.ma as ma
import funpy.model_utils as utils
import cmocean.cm as cmo 

plt.ion()
plt.style.use('ggplot')

rootdir = os.path.join('/data2','enuss','funwave_lab_setup','mono_test_ata','postprocessing')
savedir = os.path.join(rootdir, 'plots')

depFile = os.path.join(rootdir, 'compiled_output','dep.out')
dep = np.loadtxt(depFile)
[n,m] = dep.shape

## load masked eta and compute Hs
maskfile = os.path.join(rootdir, 'compiled_output', 'mask.nc')
etafile = os.path.join(rootdir, 'compiled_output', 'eta.nc')
eta, x, y = utils.load_masked_variable(os.path.join(rootdir, 'compiled_output'), 'eta', etafile, 'mask', maskfile)
Hs = utils.compute_Hsig(eta, 500, 1000)
Hs_alongmean = np.nanmean(Hs, axis=0)
Hs_alongstd = np.nanstd(Hs, axis=0)

### plot Hs
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(x, Hs_alongmean, color='tab:blue', label='Mean $H_s$ (m)')
ax1 = ax.twinx()
ax1.plot(x, -dep[0,:], color='grey')
#ax.plot(x, Hs_alongmean - Hs_alongstd, color='grey', alpha=0.6)
#ax.plot(x, Hs_alongmean + Hs_alongstd, color='grey', alpha=0.6)
#ax.fill_between(x, Hs_alongmean - Hs_alongstd, Hs_alongmean + Hs_alongstd, color='grey', alpha=0.3, label='+/- $\sigma$')
ax.set_xlabel('x (m)')
ax.set_ylabel('$H_s$')
ax.legend(loc='best')
fig.savefig(os.path.join(savedir,'hsig_alongmean_wbathy.png'))

## load masked vorticity
vortmaskfile = os.path.join(rootdir, 'compiled_output', 'vorticity_mask.nc')
vortfile = os.path.join(rootdir, 'compiled_output', 'vorticity.nc')
vort, x, y = utils.load_masked_variable(os.path.join(rootdir, 'compiled_output'), 'vorticity', vortfile, 'vorticity_mask', vortmaskfile)
[xx, yy] = np.meshgrid(x, y)

## plot planar plot of vorticity
fig, ax = plt.subplots()
p = ax.pcolormesh(xx, yy, vort[-1,:,:], cmap=cmo.curl)
p.set_clim(-0.07, 0.07)
#ax.set_aspect('equal', 'box')
fig.colorbar(p, ax=ax, label=r'$\zeta$ (s$^{-1}$)')
ax.set_xlabel('Cross-Shore (m)')
ax.set_ylabel('Alongshore (m)')
fig.savefig(os.path.join(savedir,'vorticity_snapshot.png'))

## load masked nubrk
maskfile = os.path.join(rootdir, 'compiled_output', 'mask.nc')
nubrkfile = os.path.join(rootdir, 'compiled_output', 'nubrk.nc')
nubrk, x, y = utils.load_masked_variable(os.path.join(rootdir, 'compiled_output'), 'nubrk', nubrkfile, 'mask', maskfile)
[xx, yy] = np.meshgrid(x, y)

## plot planar plot of nubrk
fig, ax = plt.subplots()
p = ax.pcolormesh(xx, yy, nubrk[-1,:,:], cmap=cmo.rain)
#p.set_clim(-0.07, 0.07)
#ax.set_aspect('equal', 'box')
fig.colorbar(p, ax=ax)
ax.set_xlabel('Cross-Shore (m)')
ax.set_ylabel('Alongshore (m)')
fig.savefig(os.path.join(savedir,'nubrk_snapshot.png'))

