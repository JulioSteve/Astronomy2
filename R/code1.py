################################################################"
# 
# PLEASE RUN THE CODE ONCE TO SEE ANSWERS TO QUESTION (i) AND (ii) AND OPEN 'Graphe.png' 
# TO SEE THE PLOT OF COMPUTED CURVES.
# ALSO, IN DATA/ FOLDER YOU WILL FIND THE ASCII FILES REQUIRED."
#
################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import quad

matplotlib.rcParams['lines.linewidth'] = 2

####################### PARAMETERS ############################# 
# (everything is put in S.I. units and corrected at the end for the expected units)
Q_H2O = 1e29
Q_HCN = 2e26
l0_H2O = 2.4e7
l1_OH = 1.6e8
l0_HCN = 2e7
l1_CN = 3e8
v0_H2O=1e3
v0_HCN = v0_H2O
v1_OH=1e3
v1_CN = v1_OH
R0=5e3
g = 4.5e-2 #ph/s/molecule
lmbd = 388e-9 #m

####################### MAIN CODE ############################# 
#Integrandes:
def parent(alpha, R0,l0,Q,v0,rho):
    """
    parent is the integrande of the column density N of PARENT molecule.
    We need to integrate this function over alpha.
    
    :variable alpha: angle between rho and ejection direction
    :param R0: radius of the comet nucleous
    :param l0: scalelength of the parent molecule
    :param Q: production rate of the considered molecule (molecule/s)
    :param v0: radical velocity of parent molecule
    :param rho: distance between nucleous and the line of sight
    """
    coef = Q*np.exp(R0/l0)/(2*np.pi*v0*rho)
    return coef*np.exp(-rho/(l0*np.cos(alpha)))

def daughter(alpha, R0,l0,Q,rho,v1,l1):
    """
    daughter is the integrande of the column density N of DAUGHTER molecule.
    We need to integrate this function over alpha.
    
    :param alpha: angle between rho and ejection direction
    :param R0: radius of the comet nucleous
    :param l0: scalelength of the parent molecule
    :param Q: production rate of the considered molecule (molecule/s)
    :param rho: distance between nucleous and the line of sight
    :param v1: radical velocity of daughter molecule
    :param l1: scalelength of the daughter molecule
    """
    coef = Q*l1/(2*np.pi*v1*rho*(l0-l1))
    exp1 = np.exp((R0-rho/np.cos(alpha))/l0)
    exp2 = np.exp((R0-rho/np.cos(alpha))/l1)
    return coef*(exp1-exp2)

#Projected distance:
Npoints=300 #points in the graphs, chose to have something smooth enough to the eye

Rho = np.linspace(10,100_000,Npoints)*1e3 #m - we chose to work in SI everywhere and convert only when needed.

#we can't start from rho<=5km because it is the radius of the nucleous.
#This would also make some infinite discontinuity for rho=0 due to division by 0. We thus chose to start at 10km.

#Computing integrals:
def mol_surfdens(R0,l0,l1,Q,v0,v1):
    """
    mol_surfdens is a function that integrates over alpha from 0 to pi/2 in order to get the -
    desired column density of Parent AND Daughter molecules. 
    
    :param R0: radius of the comet nucleous
    :param l0: scalelength of the parent molecule
    :param l1: scalelength of the daughter molecule
    :param Q: production rate of the considered molecule (molecule/s)
    :param v0: radical velocity of parent molecule
    :param v1: radical velocity of daughter molecule

    Outputs-
    :float Np: column density of parent molecule (molecule/cm^2).
    :float Nd: column density of daughter molecule (molecule/cm^2).
    """
    Np = np.zeros(shape=(Npoints))
    Nd = np.copy(Np)
    for i,rho in enumerate(Rho):
        yp = quad(parent, 0, np.pi/2, args=(R0,l0,Q,v0,rho))[0] #integrating
        yd = quad(daughter, 0, np.pi/2, args=(R0,l0,Q,rho,v1,l1))[0] #integrating
        Np[i] = yp*1e-4 #conversion from molecule/m^2 to molecule/cm^2
        Nd[i] = yd*1e-4 #conversion from molecule/m^2 to molecule/cm^2
    return Np,Nd

#Writing in ASCII files:
def Write_res(flag):
    Np1,Nd1 = mol_surfdens(R0,l0_H2O,l1_OH,Q_H2O,v0_H2O,v1_OH) #index 1 stands for H2O/OH
    Np2,Nd2 = mol_surfdens(R0,l0_HCN,l1_CN,Q_HCN,v0_HCN,v1_CN) #index 2 stands for HCN/CN
    if flag:
        np.savetxt("DATA/H2O.txt", Np1)
        np.savetxt("DATA/OH.txt", Nd1)
        np.savetxt("DATA/HCN.txt", Np2)
        np.savetxt("DATA/CN.txt", Nd2)
    return (Np1,Nd1,Np2,Nd2) 
    # ^ This allows us to recover data from calling this function as used in [1]
    # "flag" is just used to write or not in ASCII files, just an optimization step for your conveniance.

(N_H2O,N_OH,N_HCN,N_CN) = Write_res(True) #[1]

#Plotting function:
def plot1(plot,save):
    if (plot or save):
        RhoX = Rho*1e-3 #conversion m->km
        f,ax = plt.subplots(2,2,sharex=True,layout="constrained", figsize=(2560/300,1440/300), dpi=300)
        [axis.grid(visible=True) for axis in ax.flat]
        ax[0,0].plot(RhoX,N_H2O,color="blue")
        ax[0,0].set_title(r"H$_2$O", color="blue", fontweight="bold")
        ax[0,1].plot(RhoX,N_OH,color="purple")
        ax[0,1].set_title(r"OH",color="purple", fontweight="bold")

        ax[1,0].plot(RhoX,N_HCN,color="red")
        ax[1,0].set_title(r"HCN",color="red", fontweight="bold")
        ax[1,1].plot(RhoX,N_CN,color="orange")
        ax[1,1].set_title(r"CN",color="orange", fontweight="bold")
        f.suptitle("Surfacic distribution of molecules/radical\n"+r"with respect to projected distance $\rho$:", fontweight="bold")
        f.supxlabel(r"Projected distance $\rho$ (km)")
        f.supylabel(r"Molecules/radicals per cm$^2$")

        xticks = [10, 25000, 50000, 75000, 100000]
        xlabs = [f"{x:,.0f}" for x in xticks]

        for axis in ax.flat:
                axis.set_xticks(xticks, xlabs)
        
        if (plot==True and save==False):
            plt.show()
        elif (plot==False and save==True):
            plt.savefig("Graphe.png", dpi=300)
        else:
            print("Bad combination of flags (inputs of this function).")

# plot = True means you want the plot only with renderer, save=True means you generate a .png graph
plot1(plot=False,save=True)

#Total Number of Particules:
def TotN():
    # This function allows us to calculate by double integration of alpha and rho the -
    # Total Number of CN in the aperture of the spectrometer (radius 1,000km)

    func = lambda rho: 2*np.pi*rho*quad(daughter, 0, np.pi/2, args=(R0,l0_HCN,Q_HCN,rho,v1_CN,l1_CN))[0]
    N = quad(func, 10*1e3, 1000*1e3)[0]
    return N

N = TotN() #we save the total number of molecules in order to compute the flux received on Earth.
print(f"\n(i). Total number of CN radicals in the spectrometer aperture (rounded to 4 decimal places): {N:1.4e} radicals.")

#Flux on Earth:
def Flux(fluo, wavelength,Ntot):
    """
    Flux calculates the flux of received CN onto the Earth depending on:
    
    :param fluo: fluorescence efficiency (photon/s/molecule)
    :param wavelength: wavelength of the emission (in m)
    :param Ntot: Total number of radicals in the aperture 
    """
    h = 6.62e-27 #erg.s (Planck constant)
    dist = 1.496e13 #1 a.u. in cm (distance Earth-comet)
    c = 3e8 #m/s (speed of light)

    E = h*c/wavelength #erg (energy of one photon)

    S = 4*np.pi*dist**2 #Surface of the 3D sphere from the comet to Earth

    return Ntot*fluo*E/S #erg/s/cm^2

print(f"\n(ii). Flux received on Earth is (rounded to 4 decimal places): {Flux(g,lmbd,N):.4e} erg/s/cm^2.\n")

################################################################"
# 
# PLEASE RUN THE CODE ONCE TO SEE ANSWERS TO QUESTION (i) AND (ii) AND OPEN 'Graphe.png' 
# TO SEE THE PLOT OF COMPUTED CURVES.
# ALSO, IN DATA/ FOLDER YOU WILL FIND THE ASCII FILES REQUIRED."
#
################################################################