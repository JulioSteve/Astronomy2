import numpy as np
import matplotlib.pyplot as plt

times = np.load("times.npy")
Omega0 = np.load("Omega0.npy")
Omega1 = np.load("Omega1.npy")

# plt.plot(t, Omega0)

def find_peak(Omega):
    id = []
    for i in range(len(Omega)-1):
        if np.abs(Omega[i]-Omega[i+1]) > 2*np.pi*0.9:
            id.append(i)
    return id

Omega = Omega0
id = find_peak(Omega)
Omega_dot = []
for i in range(len(id)-1):
    sub_Omega = Omega[id[i]+1:id[i+1]+1]
    sub_times = times[id[i]+1:id[i+1]+1]
    Omega_dot += [(sub_Omega[-1] - sub_Omega[0])/(sub_times[-1]-sub_times[0]) for i in range(len(sub_Omega))]

print(f"Mean of Omega0 dot : {np.mean(Omega_dot)}")

Omega = Omega1
id = find_peak(Omega)
Omega_dot = []
for i in range(len(id)-1):
    sub_Omega = Omega[id[i]+1:id[i+1]+1]
    sub_times = times[id[i]+1:id[i+1]+1]
    Omega_dot += [(sub_Omega[-1] - sub_Omega[0])/(sub_times[-1]-sub_times[0]) for i in range(len(sub_Omega))]

print(f"Mean of Omega1 dot : {np.mean(Omega_dot)}")

plt.plot(times, Omega1)
plt.show()