import os
import numpy as np
import re
import pandas as pd


def find_pos(filepath, string):
	file = open(filepath, 'r')
	flag = 0
	index = -1 

	content = file.readlines()
	file = open(filepath, 'r')
	for line in file:
		index += 1
		if string in line:
			flag = 1
			break
	pos = float(re.findall(r"[-+]?\d*\.\d+|\d+", content[index])[0])
	file.close()
	return pos

def find_depth(xpos, x, dep):
	ind = np.argmin((xpos-x)**2)
	return dep[ind]

def hyd(p, dm, h0, rho=1000, g=9.81, patm=1):
	return (p-patm)/(rho*g) + dm - h0

def sl(phyd, dm, h0, dt, g=9.81):
	dsldt = pd.Series(np.gradient(phyd)/dt).rolling(20,center=True).mean().to_numpy()
	dsl2dt2 = pd.Series(np.gradient(dsldt)/dt).rolling(20,center=True).mean().to_numpy()
	return phyd - h0/(2*g)*(1 - (dm/h0)**2)*dsl2dt2

def snl(psl, dm, h0, dt, g=9.81):
	dsldt = pd.Series(np.gradient(psl)/dt).rolling(20,center=True).mean().to_numpy()
	dsl2dt2 = pd.Series(np.gradient(dsldt)/dt).rolling(20,center=True).mean().to_numpy()
	return psl - 1/g*(dsldt*dsldt + psl*dsl2dt2 - (dm/h0)**2*dsldt**2)

def load_press_insitu(filepath, duration=45, rho=1000, g=9.81):
	dat = np.loadtxt(filepath, comments='%')
	# find sensor positions
	xpos = find_pos(filepath, 'X:')
	ypos = find_pos(filepath, 'Y:')
	zpos = find_pos(filepath, 'Z:')
	dt = duration/len(dat)
	time = np.arange(0, 45, dt)*60
	return time, dat, xpos, ypos, zpos