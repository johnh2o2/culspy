import numpy as np
from math import cos

Nlcs = 5000
Npoints = 10000
Tbase = 7 * 365.
fr = 1.
amp = 1.0
sigma = 0.5

outroot="lcs/fake_%05d_N%d.dat"

def fudge(scale=0.1):
	return 1 + np.random.normal(scale=scale)

def randind(N):
	return int(N * np.random.random())

def signal(t, components, sigma=sigma):
	s = 0.0
	for c in components:
		s += c['amp'] * cos(2 * np.pi * (c['freq']*t - c['phase']))
	return s + np.random.normal(scale=sigma)

for i in range(Nlcs):
	print "Making fake lightcurve number ", i+1
	components = [ 
		{
			'freq' : fr * fudge(),
			'amp'  : amp,
			'phase' : 0.0
		} 
	]
	
	times = np.linspace(0,Tbase, Npoints) + np.random.normal(scale=0.1, size=Npoints)
	times = np.sort(times)
	times -= times[0]

	x = np.array([ signal(t, components) for t in times])
	x -= np.mean(x)

	fname = outroot%(i, Npoints)
	f = open(fname, 'w')
	for t, X in zip(times, x):
		f.write("%e %e\n"%(t, X))
	f.close()


