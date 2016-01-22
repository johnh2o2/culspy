import ctypes, culsp
import numpy as np
from time import time


culsp.initialize_cuda(0)

f = open("timing_results.txt", 'w')
Nvals = [ 100, 1000, 3000, 5000, 10000, 20000, 30000, 50000, 100000, 500000, 1000000 ]
Nfvals = [ 256, 1024, 4096, 32768, 65536, 131072 ]
#Nvals = [ 100, 1000 ]
#Nfvals = [ 512, 1024 ]

FREQ = 100.0
sigma = 0.1
minf, maxf = 0.0, 200.0
def gettx(Nt):
	t = np.linspace(0, 1, Nt)
	x = np.sin(2 * np.pi * FREQ * t) + sigma * np.random.normal(size=Nt)
	return list(t), list(x)
f.write("# Ndata  Nfreqs  time (s)\n");
for Nt in Nvals:
	
	t , x = gettx(Nt)
	for Nf in Nfvals:
		t0 = time()
		frq, power = culsp.LSP(t, x, Nf=Nf)
		dt = time() - t0
		msg = '%d %d %.5e'%(Nt, Nf, dt)
		print msg	
		f.write("%s\n"%msg);
f.close()

f = open("timing_results_std.txt", 'w')
for Nt in Nvals:
	t, x= gettx(Nt)
	t0 = time()
	culsp.LSP(t, x)
	dt = time() - t0

	msg = "%d %.5e"%(Nt, dt)
	print msg
	f.write("%s\n"%msg)

