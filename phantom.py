import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom

# settings
nx = 400
na = 400
theta = np.linspace(0., 180., na)
sigma = 0

# phantom
u = shepp_logan_phantom()

# sinogram
f = radon(u, theta=theta)
f_noisy = f + sigma * np.random.randn(nx,na)

# reconstruction
u_fbp = iradon(f_noisy,theta=theta)

# plot


'''
fig,ax = plt.subplots(1,2)

ax[0].imshow(u,extent=(-1,1,-1,1),vmin=0)
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].set_aspect(1)

ax[1].imshow(u_fbp,extent=(-1,1,-1,1),vmin=0)
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$y$')
ax[1].set_aspect(1)
'''
fig.tight_layout()
