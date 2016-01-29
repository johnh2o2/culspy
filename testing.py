import culspy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import cos
from time import time

culspy.initialize_cuda(0)

Npoints = 10000
Nbootstrap = 100

def fudge(scale=0.1):
	return 1 + np.random.normal(scale=scale)

def get_peaks_above_cutoff(vals, cutoff):

	peaks = []
	for i, v in enumerate(vals):
		if i == 0 or i == len(vals) - 1: continue
		if v > cutoff and v > vals[i-1] and v > vals[i+1]: peaks.append(i)

	return peaks	

def randind(N):
	return int(N * np.random.random())

def test_bootstrap(t, x):
	freqs, pows = culspy.LSP(t, x)
	
	heights = []
	for i in range(Nbootstrap):
		shuffled_x = [ x[randind(len(x))] for i in range(len(x)) ]
		f, p = culspy.LSP(t, shuffled_x)
		heights.append(max(p))

	return heights

def print_time(proc, t):
	print "%-40s: %.4e (s) [ %.4e (s) per bootstrap ]"%(proc, t, t/Nbootstrap)


def time_bootstrap(t, x):
	t0 = time()
	test_bootstrap(t, x)
	t = time() - t0
	print_time("python bootstrap", t)

	t0 = time()
	heights = culspy.LSPbootstrap(times, x, Nbootstrap=Nbootstrap, use_gpu_to_get_max=True)
	t = time() - t0

	print_time("culspy bootstrap (gpu max)", t)

	t0 = time()
	heights = culspy.LSPbootstrap(times, x, Nbootstrap=Nbootstrap, use_gpu_to_get_max=False)
	t = time() - t0

	print_time("culspy bootstrap (cpu max)", t)

def test_batch(ts, xs, minf=0., maxf=1., Nf=10000):

	Nlc = len(ts)
	def print_time_batch(proc, t):
		print "%-40s: %.4e (s) [ %.4e (s) per lightcurve ]"%(proc, t, t/Nlc)

	t0 = time()
	freqs, powers = culspy.LSPbatch(ts, xs, minf, maxf, Nf)
	t = time() - t0

	print_time_batch("culspy LSPbatch", t)

	t0 = time()
	freqs, powers = culspy.LSPstream(ts, xs, minf, maxf, Nf)
	t = time() - t0

	print_time_batch("culspy LSPstream", t)

	t0 = time()
	for T, X in zip(ts, xs):
		freqs, powers = culspy.LSP(T, X, minf=minf, maxf=maxf, Nf=Nf)
	t = time() - t0

	print_time_batch("culspy LSP (python batching)", t)


components = [ 
	{
		'freq' : 10 * fudge(),
		'amp'  : 1.0,
		'phase' : 0.0
	} ]
""",
	{
		'freq' : 35 * fudge(),
		'amp' : 0.6 * fudge(),
		'phase' : -0.2 * fudge()
	}
]
"""
sigma = 0.5

def signal(t, components):
	s = 0.0
	for c in components:
		s += c['amp'] * cos(2 * np.pi * (c['freq']*t - c['phase']))
	return s + np.random.normal(scale=sigma)

times = np.linspace(0,1, Npoints) + np.random.normal(scale=0.1, size=Npoints)
times = np.sort(times)

x = np.array([ signal(t, components) for t in times])

print "Testing the BATCH processing."
Nbatch = 1000
t_batch = []
x_batch = []
for i in range(Nbatch):
	Np = int(np.random.random() * Npoints)
	t0  = np.linspace(0,1, Np) + np.random.normal(scale=0.1, size=Np)
	x0 = np.array([ signal(t, components) for t in t0])
	x_batch.append(x0)
	t_batch.append(t0)
test_batch(t_batch, x_batch)


print "getting original lsp"

freqs, pows = culspy.LSP(times, x, minf=0.1, maxf=20, Nf=10000)

t0 = time()
print "getting bootstrapped LSP heights"
heights = culspy.LSPbootstrap(times, x, Nbootstrap=Nbootstrap,  minf=0.1, maxf=20, Nf=10000)
dt = time() - t0
print "  (%.3e s = %.3e s per bootstrap)"%(dt, dt/Nbootstrap)


sig1 = np.percentile(heights, 68.3)
sig2 = np.percentile(heights, 95.4)
sig3 = np.percentile(heights, 99.7)


f, (axraw, axlsp)  = plt.subplots(1, 2)

#axraw.errorbar(times, x, yerr=sigma, fmt='ko')
axraw.scatter(times, x, alpha=0.1, facecolor='b', marker=',')
axraw.set_xlim(min(times), max(times))

axlsp.set_yscale('log')
axlsp.set_xscale('log')
axlsp.plot(freqs, pows, color='k')

axlsp.axhline(sig2, color='k', ls=':')
axlsp.axhline(sig3, color='r', ls=':') 

f2, ax2 = plt.subplots()

ax2.hist(np.log10(heights), bins=Nbootstrap/20)
ax2.set_xlabel("${\\rm max}\\log_{10}P_{\\rm LS}(f)$")
ax2.set_ylabel("N")
print components

print "LSP peaks"
print "freq, power, power/1sig"
peaks = get_peaks_above_cutoff(pows, 5 * sig1)
for peak in peaks:
	print freqs[peak], pows[peak], pows[peak]/sig1

print np.sort(heights)[::10]
print np.sort(test_bootstrap(times, x))[::10]

time_bootstrap(times, x)

plt.show()
