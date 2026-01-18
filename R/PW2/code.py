import numpy as np
import matplotlib.pyplot as plt

######################### PARAMETERS ###########################

kB = 1/1.439 # special units for Boltzmann constant
Trot = 200 # K - rotational temperature
Tvib = 9000 # K - vibrational temperature
N = 3e10 # molecules/cm^{-2} - total column density

########################### DATA PROCESSING ####################

# Loading and reshaping energy levels: 10 vibrational levels, 17 rotational levels, 4 states (F1e, F2e, F1f, F2f)
Energ = np.loadtxt("DATA/energies.dat").reshape(10,17,4)
Proba = np.loadtxt("DATA/probas.dat")

# Extracting individual rotational energy matrices
F1e = Energ[:,:-1,0]
F2e = Energ[:,:-1,1]
F1f = Energ[:,:-1,2]
F2f = Energ[:,:-1,3]

# Splitting and reshaping transition probabilities by sequence (Delta v)
PDv2 = Proba[0:8*16,:].reshape(8,16,6) # Deltav = 2
# S={(0,2);(1,3);...;(7,9)} -> #S=8
PDv3 = Proba[8*16:(8+7)*16,:].reshape(7,16,6) # Deltav = 3
# S={(0,3);(1,4);...;(6,9)} -> #S=7
PDv4 = Proba[16*(8+7):(8+7+6)*16,:].reshape(6,16,6) # Deltav = 4
# S={(0,4);(1,5);...;(5,9)} -> #S=6
PDv5 = Proba[16*(8+7+6):(8+7+6+5)*16,:].reshape(5,16,6) # Deltav = 5
# S={(0,5);(1,6);...;(4,9)} -> #S=5

################################################################

# Computing the vibrational energy G(v) as the minimum energy of the level (non-zero)
G = np.zeros(shape=10)
for v in range(10):
    E = Energ[v,0,:]
    temp = []
    for k in range(len(E)):
        if E[k] != 0:
            temp.append(E[k])
    G[v] = min(temp)

def calc_Q(G):
    # Global vibrational partition function
    S = 0
    for v in range(10):
        S += np.exp(-1.439*G[v]/Tvib)
    return S

Q = calc_Q(G) # Computed once so we don't have to do it again

def calc_Nv(v,Q,G):
    # Population of the vibrational level v (Boltzmann distribution in specific units)
    return N*np.exp(-1.439*G[v]/Tvib)/Q

def calc_Qv(v):
    # Rotational partition function for a specific vibrational level v
    S1e = 0
    S2e = 0
    S1f = 0
    S2f = 0
    # J = 0.5 only exists for Pi_1/2 (substate 2)
    S2e += (2*0.5+1)*np.exp(-F2e[v,0]/(kB*Trot))
    S2f += (2*0.5+1)*np.exp(-F2f[v,0]/(kB*Trot))
    for j in range(1,16):
        J = j+0.5
        S2e += (2*J+1)*np.exp(-F2e[v,j]/(kB*Trot))
        S2f += (2*J+1)*np.exp(-F2f[v,j]/(kB*Trot))
        S1e += (2*J+1)*np.exp(-F1e[v,j]/(kB*Trot))
        S1f += (2*J+1)*np.exp(-F1f[v,j]/(kB*Trot))
    return S1e+S2e+S1f+S2f

def calc_I(filename="SpectrumDATA.dat"):
    # Main function to compute transition intensities and generate the output file
    # "haut" refers to ' (upper) and "bas" to '' (lower)
    all_lambdas = []
    all_intensities = []
    output_rows = []

    # Transition sequences to process:
    Proba_lists = [PDv2, PDv3, PDv4, PDv5]
    
    for i, DvMatrix in enumerate(Proba_lists):
        dv = i + 2  # dv sequence: 2, 3, 4 then 5
        
        for v_bas in range(DvMatrix.shape[0]):
            v_haut = v_bas + dv
            
            # Terms depending only on the upper vibrational level v'
            Nv_haut = calc_Nv(v_haut, Q, G)
            Qv_haut = calc_Qv(v_haut)
            pop_ratio = Nv_haut / Qv_haut
            
            for j_idx in range(16):
                J_haut = j_idx + 0.5
                
                for k in range(6):
                    A = DvMatrix[v_bas, j_idx, k]
                    if A <= 0:
                        continue
                    
                    # Identifying substate (1: 3/2 or 2: 1/2) and averaging e/f upper energy
                    substate = 1 if k < 3 else 2
                    if substate == 1: # (P1, Q1, R1)
                        E_haut = (F1e[v_haut, j_idx] + F1f[v_haut, j_idx]) / 2
                    else:             # (P2, Q2, R2)
                        E_haut = (F2e[v_haut, j_idx] + F2f[v_haut, j_idx]) / 2
                    
                    # Selection rules for J" based on branch k
                    branch_type = k % 3 # 0: P, 1: Q, 2: R
                    if branch_type == 0:   # P branch: J" = J' + 1
                        j_bas_idx = j_idx + 1
                    elif branch_type == 1: # Q branch: J" = J'
                        j_bas_idx = j_idx
                    else:                  # R branch: J" = J' - 1
                        j_bas_idx = j_idx - 1
                    
                    # Checking rotational index boundaries
                    if 0 <= j_bas_idx < 16:
                        J_bas = j_bas_idx + 0.5
                        # Averaging e/f lower energy for E"
                        if substate == 1:
                            E_bas = (F1e[v_bas, j_bas_idx] + F1f[v_bas, j_bas_idx]) / 2
                        else:
                            E_bas = (F2e[v_bas, j_bas_idx] + F2f[v_bas, j_bas_idx]) / 2
                        
                        # Calculating line position and intensity
                        dE = E_haut - E_bas
                        if dE > 0:
                            # Position: lambda(nm) = 10^7 / delta_E(cm^-1)
                            wavelength_nm = 1e7 / dE
                            
                            # Intensity in Rayleigh: I = 10^-6 * N(upper) * A
                            exponent = np.exp(-1.439 * E_haut / Trot)
                            intensity = 1e-6 * pop_ratio * (2 * J_haut + 1) * exponent * A
                            
                            all_lambdas.append(wavelength_nm)
                            all_intensities.append(intensity)

                            # Saving data for ASCII output
                            output_rows.append([
                                wavelength_nm, intensity, v_haut, v_bas, 
                                substate, J_haut, J_bas
                            ])
                            
# Exporting results to ASCII file with 7 columns
    header = "Wavelength(nm)\tIntensity(Rayleigh)\tv'\tv''\tSubstate\tJ'\tJ''"
    np.savetxt(filename, output_rows, fmt='%.4f\t%.4e\t%d\t%d\t%d\t%.1f\t%.1f', header=header)
    
    return np.array(all_lambdas), np.array(all_intensities)


### MAIN EXECUTION ###
Nv = np.array([calc_Nv(v,Q,G) for v in range(10)])
Qv = np.array([calc_Qv(v) for v in range(10)])

# Compute the spectrum and save the data
lambdas, intensities = calc_I()

# Plotting the discrete emission lines (stick spectrum)
plt.figure(figsize=(2560/300, 1440/300))
plt.vlines(lambdas, 0, intensities, colors='blue', linewidth=1)
plt.xlabel('Wavelength(nm)')
plt.ylabel('Intensity (Rayleigh)')
plt.title(r'Emission spectrum of OH ($\Delta v\in[|2;5|]$ )')
plt.grid(True, alpha=0.3)
plt.savefig("Spectrum.png", format="png", dpi=300)