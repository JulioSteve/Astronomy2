import numpy as np
import matplotlib.pyplot as plt

######################### PARAMETERS ###########################

kB = 1/1.439 #special units
Trot = 200 #K
Tvib = 9000 #K
N = 3e10 #molecules/cm^{-2}

########################### DATA PROCESSING ####################

Energ = np.loadtxt("DATA/energies.dat").reshape(10,17,4)
Proba = np.loadtxt("DATA/probas.dat")

F1e = Energ[:,:,0]
F2e = Energ[:,:,1]
F1f = Energ[:,:,2]
F2f = Energ[:,:,3]

PDv2 = Proba[0:8*16,:].reshape(8,16,6) #Deltav = 2
# S={(0,2);(1,3);...;(7,9)} -> #S=8
PDv3 = Proba[8*16:(8+7)*16,:].reshape(7,16,6) #Deltav = 3
# S={(0,3);(1,4);...;(6,9)} -> #S=7
PDv4 = Proba[16*(8+7):(8+7+6)*16,:].reshape(6,16,6) #Deltav = 4
# S={(0,4);(1,5);...;(5,9)} -> #S=6
PDv5 = Proba[16*(8+7+6):(8+7+6+5)*16,:].reshape(5,16,6) #Deltav = 5
# S={(0,5);(1,6);...;(4,9)} -> #S=5

################################################################

G = np.zeros(shape=10)
for v in range(10):
    E = Energ[v,0,:]
    temp = []
    for k in range(len(E)):
        if E[k] != 0:
            temp.append(E[k])
    G[v] = min(temp)

def calc_Q(G):
    S = 0
    for v in range(10):
        S += np.exp(-1.439*G[v]/Tvib)
    return S

def calc_Nv(v,Q,G):
    return N*np.exp(-1.439*G[v]/Tvib)/Q

def calc_Qv(v):
    S1e = 0
    S2e = 0
    S1f = 0
    S2f = 0
    S2e += (2*0.5+1)*np.exp(-F2e[v,0]/(kB*Trot))
    S2f += (2*0.5+1)*np.exp(-F2f[v,0]/(kB*Trot))
    for J in range(1,16):
        j = J+0.5
        S2e += (2*j+1)*np.exp(-F2e[v,J]/(kB*Trot))
        S2f += (2*j+1)*np.exp(-F2f[v,J]/(kB*Trot))
        S1e += (2*j+1)*np.exp(-F1e[v,J]/(kB*Trot))
        S1f += (2*j+1)*np.exp(-F1f[v,J]/(kB*Trot))
    return S1e+S2e+S1f+S2f

def calc_I(L):
    IDv2 = np.zeros(shape=(8,16)) #Dv = 2 -> contains intensities for each v' and each J' with Dv=2
    IDv3 = np.zeros(shape=(7,16)) #Dv = 3
    IDv4 = np.zeros(shape=(6,16)) #Dv = 4
    IDv5 = np.zeros(shape=(5,16)) #Dv = 5

    



### Using previous functions:
Q = calc_Q(G)
Nv = np.array([calc_Nv(v,Q,G) for v in range(10)])
Qv = np.array([calc_Qv(v) for v in range(10)])
