# Plot the crossover at which the terms in 1-Psi(x|b) become monotonically decreasing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

xs = np.linspace(0.1, 20.0)
bs = np.linspace(0.01, 1.0)
ns = np.arange(0,10)

thr = np.zeros((xs.size, bs.size))

for i,x in enumerate(xs):
    for j, b in enumerate(bs):
        ell = (2*ns**2 + ns*b + 2*ns + b + 2*(ns+1)*b + b**2) / (2*ns**2 + ns*b + 2*ns + b)
        r = np.exp(-(4*ns+2*b + 2)/x)
        thr[i,j] = np.amin(np.where(ell*r < 1))
        print "x: ", x, " b: ", b, " n: ", thr[i,j]
    
        

plt.figure(figsize=(3,3))

plt.imshow(thr, cmap="YlOrRd", interpolation="none", extent=(bs[0], bs[-1], xs[-1], xs[0]), aspect=bs[-1]/xs[-1])
plt.text(0.15, 1.5, "$\\theta_n=0$")
plt.text(0.4, 10.0, "$\\theta_n=1$")
plt.text(0.65, 18.5, "$\\theta_n=2$")
plt.xlabel("$b$")
plt.ylabel("$x$")
#plt.colorbar()
plt.tight_layout()
plt.savefig("n_threshold.pdf")
plt.show()
