import eduviges.paramo as para
import numpy as np
from Eduviges import constants as econs
import plots_code as pc

##Example injectecting a broken power law into a blob that is cooled by synchrotron and ssc



###constants
numt = 200
numg = 80
numf = 80
fmax = 1e28
fmin =1e8
tmax = 1e18
gmin = 1e1
gmax = 1e8
with_abs = True
cool_withKN = False

R = 1e14 # size of blob
sigma = 1e-6 #blobs magnetization
B=0.1 #blobs magnetic field
n0 = (B**2)/(np.pi*4*sigma*econs.mp*(econs.cLight**2))#particle density
t_esc = R/econs.cLight #escape time
t_inj = R/econs.cLight #injection time
tlc = R/econs.cLight #light crossing time
p1 = -2.5 #pwl indices
p2 =p1 -1
gcut = 1e2
g2cut = 1e5

tmax = tlc*2

###arrays
n = np.zeros([numt,numg]) #particle distribution dn/d\gamma
gdot = np.zeros([numt,numg]) #fp cooling term
j_s = np.zeros([numt,numf]) #synchrotron emissivity
j_ssc = np.zeros([numt,numf]) #ssc emissivity
I_s = np.zeros([numt,numf])#synchrotron Intensity
I_ssc = np.zeros([numt,numf])#ssc Intensity
ambs = np.zeros([numt,numf]) #absorbtion coefficient
t = np.logspace(0,np.log10(tmax),numt) # logspaced time array where t_0 = 1
f = np.logspace(np.log10(fmin),np.log10(fmax),numf) # logspaced time array where t_0 = 1
g = np.logspace(np.log10(gmin),np.log10(gmax),numg) #logspaced lorentz factor array
D = np.full(numg,1e-200) # diffusion array
gdot[0,:] = np.full(numg,1e-200) #cooling array
Qinj = np.zeros(numg) #injection distribution


def broken_pwl(n0,g,p1,p2,gmin_cut,g2_cut):
    f = np.zeros(len(g))
    dg = np.zeros(len(g))
    i0 = None
    g2cut =True
    for i in range(len(g)):
        if i >0:
            dg[i] = g[i]-g[i-1]
        if(g[i]>gmin_cut and g[i]<g2_cut):
            f[i] = (g[i])**p1
        elif(g[i]>g2_cut):
            if(g2cut):
                g2cut=False
                i0=i-1
            f[i] = f[i0]*((g[i]/g[i0])**p2)
        else:
            f[i] = 0
    dg[0] = dg[1]
    f = n0*f/sum(dg*f) #very rough normalization
    return f

##define fp terms
Qinj = broken_pwl(n0,g,p1,p2,gcut,g2cut)/t_inj
gdot[0,:] = (4/3)*econs.sigmaT*econs.cLight*((B**2)/(8*np.pi))*(g**2)/(econs.me*(econs.cLight**2))
n[0,:] = broken_pwl(n0,g,p1,p2,gcut,g2cut)
###time loop
for i in range(1,len(t)):
    dt = t[i] - t[i-1]

    n[i,:] = para.distribs.fp_findif_difu(dt, g, n[i-1,:], gdot[i-1,:], D, Qinj, t_esc, tlc)
    j_s[i,:],ambs[i,:] = para.radiation.syn_emissivity_full(f,g,n[i,:],B,with_abs) #,sync and absorb
    I_s[i,:] = para.radiation.radtrans_blob(j_s[i,:],R,ambs[i,:])
    j_ssc[i,:] = para.radiation.ic_iso_powlaw_full(f,I_s[i,:],g,n[i,:])
    I_ssc[i, :] = para.radiation.radtrans_blob(j_ssc[i, :], R, ambs[i, :])
    dotgKN = para.radiation.rad_cool_pwl(g, f, 4 * np.pi * I_ssc[i, :]  / econs.cLight, cool_withKN)
    gdot[i,:] = gdot[0,:] + dotgKN

pc.plot_n(g,n,t)
pc.plot_j(f,f*(j_s+j_ssc),t)
pc.plot_I(f,np.pi * 4* (I_ssc + I_s)*f,t)