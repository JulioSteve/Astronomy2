import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import quad, dblquad

matplotlib.rcParams['lines.linewidth'] = 3

### PARAMETERS (everything is put in S.I. units and corrected at the end for the expected units)
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
g = 4.5e-2 #ph/s
lmbd = 388e-9 #nm
################

### MAIN CODE
#Integrandes
def parent(alpha, R0,l0,Q,v0,rho):
    coef = Q*np.exp(R0/l0)/(2*np.pi*v0*rho)
    return coef*np.exp(-rho/(l0*np.cos(alpha)))

def daughter(alpha, R0,l0,Q,rho,v1,l1):
    coef = Q*l1/(2*np.pi*v1*rho*(l0-l1))
    exp1 = np.exp((R0-rho/np.cos(alpha))/l0)
    exp2 = np.exp((R0-rho/np.cos(alpha))/l1)
    return coef*(exp1-exp2)

#Projected distance
Npoints=100
Rho = np.linspace(10,100_000,Npoints)*1e3

#Computing integrals
def mol_surfdens(R0,l0,l1,Q,v0,v1):
    Np = np.zeros(shape=(Npoints))
    Nd = np.copy(Np)
    for i,rho in enumerate(Rho):
        yp = quad(parent, 0, np.pi/2, args=(R0,l0,Q,v0,rho))[0]
        yd = quad(daughter, 0, np.pi/2, args=(R0,l0,Q,rho,v1,l1))[0]
        Np[i] = yp*1e-4
        Nd[i] = yd*1e-4
    return Np,Nd

#Writing in ASCII files
def Write_res(flag):
    Np1,Nd1 = mol_surfdens(R0,l0_H2O,l1_OH,Q_H2O,v0_H2O,v1_OH)
    Np2,Nd2 = mol_surfdens(R0,l0_HCN,l1_CN,Q_HCN,v0_HCN,v1_CN)
    if flag:
        np.savetxt("DATA/H2O.txt", Np1)
        np.savetxt("DATA/OH.txt", Nd1)
        np.savetxt("DATA/HCN.txt", Np2)
        np.savetxt("DATA/CN.txt", Nd2)
    return (Np1,Nd1,Np2,Nd2)

(N_H2O,N_OH,N_HCN,N_CN) = Write_res(True)

#Plotting
def plot1(flag,save):
    if flag:
        RhoX = Rho*1e-3
        f,ax = plt.subplots(2,2,sharex=True,layout="constrained")
        ax[0,0].plot(RhoX,N_H2O,color="blue")
        ax[0,0].set_title(r"H$_2$O")
        ax[0,1].plot(RhoX,N_OH,color="purple")
        ax[0,1].set_title(r"OH")

        ax[1,0].plot(RhoX,N_HCN,color="red")
        ax[1,0].set_title(r"HCN")
        ax[1,1].plot(RhoX,N_CN,color="orange")
        ax[1,1].set_title(r"CN")
        f.suptitle(r"Surfacic distribution of molecules/radicals with respect to projected distance $\rho$:")
        f.supxlabel(r"Projected distance $\rho$ (km)")
        f.supylabel(r"Molecules/radicals per cm$^2$")
        
        if save!=True:
            plt.show()
        else:
            plt.savefig("Graphe1.png")

plot1(True,save=True)

#Flux
def TotN():
    func = lambda rho: 2*np.pi*rho*quad(daughter, 0, np.pi/2, args=(R0,l0_HCN,Q_HCN,rho,v1_CN,l1_CN))[0]
    N = quad(func, 10*1e3, 1000*1e3)[0]
    return N

N = TotN()
print(f"(i). Total number of CN radicals in the spectrometer aperture (approximately): {N:1.4e} radicals.")