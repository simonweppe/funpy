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
    flist = sorted(flist)
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

def uv2vorticity(fdir, savefile = 'vorticity.nc', ufile = 'u.nc', vfile = 'v.nc'):
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

    u = np.asarray(u_dat['u'])

    v_dat = xr.open_dataset(os.path.join(fdir, vfile))
    v = np.asarray(v_dat['v'])

    dvdx = np.gradient(v, dx, axis=2)
    dudy = np.gradient(u, dy, axis=1)

    vor = dvdx - dudy

    dim = ["time", "y", "x"]
    coords = [time, y, x]
    dat = xr.DataArray(vor, coords=coords, dims=dim, name='vorticity')
    dat.to_netcdf(os.path.join(fdir, savefile))

def vorticity2netcdf(fdir, nchunks=1):
    for i in range(nchunks):
        savefile = 'vorticity_%d.nc' % i
        ufile = 'u_%d.nc' % i
        vfile = 'v_%d.nc' % i
        uv2vorticity(fdir, savefile, ufile, vfile)


def compute_fbr(vel, name, fdir, savefile='fbr.nc', nufile='nubrk.nc', etafile='eta.nc', depfile='dep.out', dx=0.05, dy=0.1, dt=0.2):
    dudx = np.gradient(vel, dx, axis=2)
    dudy = np.gradient(vel, dy, axis=1)
    del vel 
    
    eta_dat = xr.open_dataset(os.path.join(fdir, etafile))
    dep = np.loadtxt(os.path.join(fdir, depfile))
    eta = eta_dat['eta']
    x = eta_dat['x']
    y = eta_dat['y']    
    heta = np.asarray([dep + eta[i,:,:] for i in range(len(eta))])  
    del eta, dep
    
    nubrk_dat = xr.open_dataset(os.path.join(fdir, nufile)) 
    nubrk = nubrk_dat['nubrk']

    term1 = nubrk * heta * dudx 
    term2 = nubrk * heta * dudy

    del nubrk

    term1dx = np.gradient(term1, dx, axis=2)
    term2dy = np.gradient(term2, dy, axis=1)
    del term1, term2
    
    fbr = 1/heta * (term1dx + term2dy)  

    T = len(fbr)
    dim = ["time", "y", "x"]
    coords = [np.linspace(0,T*dt,T), y, x]
    dat = xr.DataArray(fbr, coords=coords, dims=dim, name=name)
    dat.to_netcdf(os.path.join(fdir, savefile))

def fbr2netcdf(fdir, nchunks=1, dx=0.05, dy=0.1, dt=0.2):
    for i in range(nchunks):
        usavefile = 'fbrx_%d.nc' % i 
        vsavefile = 'fbry_%d.nc' % i 
        ufile = 'u_%d.nc' % i
        vfile = 'v_%d.nc' % i
        nufile = 'nubrk_%d.nc' % i 
        etafile = 'eta_%d.nc' % i 
        u = xr.open_dataset(os.path.join(fdir, ufile))['u']
        compute_fbr(u, 'fbrx', fdir, usavefile, nufile, etafile)
        del u 
        v = xr.open_dataset(os.path.join(fdir, vfile))['v']
        compute_fbr(v, 'fbry', fdir, vsavefile, nufile, etafile)
        del v 


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

def crest2netcdf(fdir, nchunks=1, threshold=0, dt=0.2):
    for i in range(nchunks):
        savefile = 'crest_%d.nc' % i
        nufile = 'nubrk_%d.nc' % i 
        maskfile = 'mask_%d.nc' % i 
        crest_identification(fdir, nufile=nufile, maskfile=maskfile, savefile=savefile)

def veldec2netcdf(savedir, nchunks=1, dx=0.05, dy=0.1, dt=0.2):
    for i in range(nchunks):
        upsifile = 'u_psi_%d.nc' % i
        vpsifile = 'v_psi_%d.nc' % i
        uphifile = 'u_phi_%d.nc' % i
        vphifile = 'v_phi_%d.nc' % i    
        u = xr.open_dataset(os.path.join(savedir, 'u_%d.nc' % i))['u']
        v = xr.open_dataset(os.path.join(savedir, 'v_%d.nc' % i))['v']  
        x = xr.open_dataset(os.path.join(savedir, 'u_%d.nc' % i))['x']
        y = xr.open_dataset(os.path.join(savedir, 'u_%d.nc' % i))['y']        
        u_psi, v_psi, u_phi, v_phi = mod_utils.vel_decomposition(u, v, dx, dy)
        var_to_netcdf(u_psi, x, y, dt, 'u_psi', os.path.join(savedir, upsifile))
        del u_psi 
        var_to_netcdf(v_psi, x, y, dt, 'v_psi', os.path.join(savedir, vpsifile))
        del v_psi
        var_to_netcdf(u_phi, x, y, dt, 'u_phi', os.path.join(savedir, uphifile))
        del u_phi 
        var_to_netcdf(v_phi, x, y, dt, 'v_phi', os.path.join(savedir, vphifile))
        del v_phi







