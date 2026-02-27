import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import os
plt.get_backend()



'''
From the documentation of Healpy and tutorial, we collect power spectrum from Plank using Url.
There are different types of spectra available, which one is best to use??

I chose one from theory - 


# get the Plank power spectrum
url = 'https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt'
#os.system(f'wget {url}')
'''



input_cl2 = np.loadtxt("./COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt")

# The input power spectrum is in text format, the 1st index = ell, 2nd = TT, 3rd = TE, 4th = EE, BB ,6TH = PP 


print(input_cl2.shape)



# ploting temperature the power spectrum
ell = input_cl2[:,0]
lmax = np.max(ell)
cl = input_cl2

# make sure dimensions are the same
ell = np.arange(2, 2509)


plt.figure()
plt.plot(ell, cl[:,1])
plt.xlabel('$\ell$')
plt.ylabel('$\ell (\ell + 1) c_ell$')
plt.grid(True)
#plt.savefig('image1_cl.png')
plt.show()


'''
In order to find out if the data is plotted as c_ell or ell(ell + 1) / 2pi, we can compare with the Temperature auto correlation - TT
'''


'''

_________________________________________________________________________________________________________________________________________________________________





Generating artificial CMB map:

1. We use the syntfast function from healpy, but this generate different realization/outcome everytime we run the func

2. We can however use the spherical harminics a_ell,m - and transform it back to a map for the realization to be the same
'''

# create a random seed
np.random.seed(seed= 42)


# create the a_el,m
cl_tt = cl[:,1] * 2 * np.pi / (ell * (ell + 1))

alm = hp.synalm(cl_tt)

# define nside
Nside = 1024
cmb_map = hp.alm2map(alm, nside= Nside)          #  lmax= lmax

plt.figure()
hp.mollview(cmb_map)
#           min= -300*1e-6, max= 300*1e-6,
#           cmap= 'jet'    
#           )
#plt.savefig('artificial_map')
#hp.gradicule()
plt.show()



# for this artificial map we can then make a power spectrum out of it - by converting the map back to alm
#cl_tt_check = hp.map2









##fnl = 1.0         
##
##map_NG = cmb_map + fnl * (cmb_map**2)
##
##
### converting the map back to alm -- but isnt that changing the appearance of the initial map????
### The cosmic variance
##cl_NG = hp.anafast(map_NG)
##
##plt.plot(ell, cl_NG)
##
##plt.savefig('2.png')
### filtering the maps in harmonic space - but. why???
### how do we choose filter weights??
### From the paper - Wiener filter?
##
##filtered_vector = alm * 0
##
##
##'''
#
#
#
