import rebound
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['legend.fontsize'] = 17
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
figsize=(14,6)

sim = rebound.Simulation("DATA.bin")

# We can add Saturn and its moons by name, since REBOUND is linked to the HORIZONS database.
labels = ["699","Mimas","Tethys","Titan", "Enceladus", "Dione", "Rhea"]
# labels = ["699","Mimas","Tethys","Titan"]

os = sim.orbits()
print("Planet: Saturn and moons:",", ".join(f"{labels[i]}" for i in range(1,len(labels))))
n_values = [os[i].n for i in range(len(labels)-1)]
P_values = [os[i].P for i in range(len(labels)-1)]

print(f"n_i (in rad/days) = ",", ".join(f"{n:.3f}" for n in n_values))
print(f"P_i (in days) = ",", ".join(f"{P:.3f}" for P in P_values))

sim.move_to_com()
def plot_orbs(flag):
    if flag:
        fig, ax = plt.subplots(figsize=(20,20))
        rebound.OrbitPlot(sim, unitlabel="[AU]", color=True, periastron=True, fig=fig, ax=ax, lw=10)
        l = 0.009
        ax.set_xlim(-l,l)
        ax.set_ylim(-l,l)

        ax.set_axis_off()
        plt.show()
        # plt.savefig("pagedegarde.png", format="png", dpi=300)

sim.integrator = "whfast"
dt = 0.05
sim.dt = dt * os[0].P  # 5% of Mimas period
Nout = 100_000            # number of points to display
tmax = 80*365.25         # let the simulation run for 80 years
Nmoons = len(labels)-1

x = np.zeros((Nmoons,Nout))
ecc = np.zeros((Nmoons,Nout))
longitude = np.zeros((Nmoons,Nout))
Omega = np.zeros((Nmoons,Nout))
n = np.zeros((Nmoons,Nout))
varpi = np.zeros((Nmoons,Nout))
pos = np.zeros(shape=(len(labels),3,Nout))
vit = np.zeros(shape=(len(labels),3,Nout))

times = np.linspace(0.,tmax,Nout)
ps = sim.particles

def ForceJ2(reb_sim):
    J2 = 16298e-6
    RSat = 0.00038925688 #AU
    MSat = 0.00028588598 #Msun
    for sat in range(1,Nmoons+1):
        prx = ps[sat].x-ps[0].x
        pry = ps[sat].y-ps[0].y
        prz = ps[sat].z-ps[0].z
        pr2 = prx*prx + pry*pry + prz*prz
        fac = -3*sim.G*J2*MSat*(RSat**2)/2/(pr2**(3.5))
        
        pax = fac*prx*(prx*prx + pry*pry - 4*prz*prz)
        pay = fac*pry*(prx*prx + pry*pry - 4*prz*prz)
        paz = fac*prz*(3*(prx*prx + pry*pry) - 2*prz*prz)

        ps[sat].ax += pax
        ps[sat].ay += pay
        ps[sat].az += paz

        mfac = ps[sat].m/ps[0].m

        ps[0].ax -= mfac*pax
        ps[0].ay -= mfac*pay
        ps[0].az -= mfac*paz

### UNCOMMENT TO ADD J2 EFFECTS TO THE SIMULATION
sim.additional_forces = ForceJ2

for i,time in enumerate(times):
    sim.integrate(time)
    # note we use integrate() with the default exact_finish_time=1, which changes the timestep near 
    # the outputs to match the output times we want.  This is what we want for a Fourier spectrum, 
    # but technically breaks WHFast's symplectic nature.  Not a big deal here.
    os = sim.orbits()
    for k,body in enumerate(ps):
        pos[k,0,i] = ps[k].x
        pos[k,1,i] = ps[k].y
        pos[k,2,i] = ps[k].z

        vit[k,0,i] = ps[k].vx
        vit[k,1,i] = ps[k].vy
        vit[k,2,i] = ps[k].vz

    for j in range(Nmoons):
        x[j][i] = ps[j+1].x 
        ecc[j][i] = os[j].e
        longitude[j][i] = os[j].l
        Omega[j][i] = os[j].Omega
        n[j][i] = os[j].n
        varpi[j][i] = os[j].pomega

def plot_Omega(flag):
    if flag:
        fig, ax = plt.subplots(2, layout="constrained", figsize=figsize)

        ax[0].plot(times,Omega[0], color="blue")
        ax[0].set_title("Mimas", color="blue", fontweight='bold')
        ax[1].plot(times,Omega[1], color="red")
        ax[1].set_title("Tethys", color="red", fontweight='bold')

        fig.supxlabel(r"Time (days)")
        fig.supylabel(r"Longitude of the ascending node (rad)")

        plt.show()
        # plt.savefig("Omegas/OmegasNoJ2.png", format="png")

def plot_n(flag):
    if flag:
        fig, ax = plt.subplots(2, layout="constrained", figsize=figsize)

        ax[0].plot(times,n[0], color="blue")
        ax[0].set_title("Mimas", color="blue", fontweight='bold')
        ax[1].plot(times,n[1], color="red")
        ax[1].set_title("Tethys", color="red", fontweight='bold')

        fig.supxlabel(r"Time (days)")
        fig.supylabel(r"Mean motion (rad/day)")

        plt.show()
        # plt.savefig("ns/nsJ2.png", format="png")

def plot_varpi(flag):
    if flag:
        fig, ax = plt.subplots(2, layout="constrained", figsize=figsize)

        ax[0].plot(times,varpi[0], color="blue")
        ax[0].set_title("Mimas", color="blue", fontweight='bold')
        ax[1].plot(times,varpi[1], color="red")
        ax[1].set_title("Tethys", color="red", fontweight='bold')

        fig.supxlabel(r"Time (days)")
        fig.supylabel(r"Longitude of the pericenter (rad)")

        plt.show()
        # plt.savefig("varpi/varpiJ2.png", format="png")


def plot_ecc(flag):
    if flag:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        plt.plot(times,ecc[0],label=labels[1])
        plt.plot(times,ecc[1],label=labels[2])
        plt.plot(times,ecc[2],label=labels[3])
        # plt.plot(times,ecc[3],label=labels[4])
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Eccentricity")
        plt.legend()
        plt.show()

def plot_xt(flag):
    if flag:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        plt.plot(times,x[0],label=labels[1])
        plt.plot(times,x[1],label=labels[2])
        plt.plot(times,x[2],label=labels[3])
        # plt.plot(times,x[3],label=labels[4])
        ax.set_xlim(0,0.2*365.25)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("x locations (AU)")
        ax.tick_params()
        plt.legend()

# def zeroTo360(val):
#     while val < 0:
#         val += 2*np.pi
#     while val > 2*np.pi:
#         val -= 2*np.pi
#     return (val*180/np.pi)

# def min180To180(val):
#     while val < -np.pi:
#         val += 2*np.pi
#     while val > np.pi:
#         val -= 2*np.pi
#     return (val*180/np.pi)

# We can calculate theta, the resonant argument of the 1:2 Mimas-Thetys orbital resonance,
# which oscillates about 0 degrees:
# theta = [min180To180(2.*longitude[0][i] - 4.*longitude[1][i] + Omega[0][i] + Omega[1][i]) for i in range(Nout)]
theta2 = [2.*longitude[0][i] - 4.*longitude[1][i] + Omega[0][i] + Omega[1][i] for i in range(Nout)]

# There is also a secular resonance argument, corresponding to the difference in the longitude of perihelions:
# This angle oscillates around 180 degs, with a longer period component.
# theta_sec = [zeroTo360(-varpi[1][i] + varpi[0][i]) for i in range(Nout)]

def plot_resonant(flag):
    if flag:
        fig,ax = plt.subplots(1,2, figsize=figsize, layout="constrained")
        ax[0].plot(times,theta2, color="darkorange")  
        ax[0].set_title("Complete Simulation")
        ax[1].set_title("Zoom over a 30-day span")
        ax[1].plot(times,theta2, color="darkorange")
        ax[1].set_xlim(9265,9295)
        # ax.plot(times,theta_sec, color="royalblue") # secular resonance argument
        # ax.set_xlim([0,20.*365.25])
        # ax.set_xlim([10*365.25,11*365.25])
        # ax.set_ylim([-180,360.])
        fig.supxlabel("Time (days)")
        fig.supylabel("Resonant argument Mimas-Tethys (rad)")
        # ax.plot([0,100],[180,180],'k--')
        # ax.plot([0,100],[0,0],'k--')
        plt.show()
        # plt.savefig("phi/6moonsphiJ2.png", format="png")

def dOmega_dt(times, Omega):
    # Déroule la phase pour éviter les sauts 2π
    Omega_unwrapped = np.unwrap(Omega)
    # Pente moyenne globale
    return np.gradient(Omega_unwrapped,times)



Omega0_dot = dOmega_dt(times, Omega[0])
Omega1_dot = dOmega_dt(times, Omega[1])

# fig, ax = plt.subplots(2, layout="constrained")
# fig.suptitle("Omega TEST")
# ax[0].plot(times,np.unwrap(Omega)[0])
# ax[1].plot(times,np.unwrap(Omega)[1])
# plt.show()

# print(f"\nMean nodal precession rate (Mimas): <dOmega0/dt> = {np.mean(Omega0_dot):.3f}")
# print(f"Mean nodal precession rate (Thetys): <dOmega1/dt> = {np.mean(Omega1_dot):.3f}")

# np.save("n0", n[0])
# np.save("n1", n[1])
# np.save("Omega0", Omega[0])
# np.save("Omega1", Omega[1])
# np.save("times", times)

def plot_OmegaDot(flag):
    if flag:
        fig, ax = plt.subplots(2, layout="constrained", figsize=figsize)
        fig.suptitle(r'$\frac{d \Omega}{dt}(t)$:')

        ax[0].plot(times, Omega0_dot)
        ax[1].plot(times, Omega0_dot)

        plt.show()


phi_dot_bienmieuxbetter = 2*n[0]-4*n[1]+Omega0_dot+Omega1_dot
def plot_dres_dt(flag):
    if flag:
        plt.figure(figsize=figsize)
        plt.plot(times,phi_dot_bienmieuxbetter, color="darkorange")
        # plt.plot(times,n[1])
        # plt.show()
        plt.savefig("phidot/6moons_phidot_J2.png", format="png")

print(f"\nMean resonant angle rate: <dphi/dt> = {np.mean(phi_dot_bienmieuxbetter):.5f}")


# dO = (np.gradient(Omega[0], times),np.gradient(Omega[1], times))
# phidot = 2*n[0]-4*n[1]+dO[0]+dO[1]
# def plot_gradresangle(flag):
#     if flag:          
#         plt.plot(times,phidot, lw=4, color="red")
#         plt.show()
# print(f"UNWRAPPED + GRADIANT: <dphi/dt> = {np.mean(phidot):.5f}")


Ekin = 0
for k in range(len(labels)):
    vx = vit[k,0]
    vy = vit[k,1]
    vz = vit[k,2]
    m = ps[k].m
    Ekin += 0.5*m*(vx*vx+vy*vy+vz*vz)

Epot = 0
for i in range(len(ps)):
    for j in range(len(ps)):
        if i!=j:
            m1 = ps[i].m
            m2 = ps[j].m
            
            x1 = pos[i,0]
            y1 = pos[i,1]
            z1 = pos[i,2]

            x2 = pos[j,0]
            y2 = pos[j,1]
            z2 = pos[j,2]

            r = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            Epot -= sim.G*m1*m2/r

Etot = Ekin+Epot

def plot_Energ(flag):
    if flag:
        plt.figure(figsize=figsize, layout="constrained")
        plt.title("Using the six moons around Saturn\n"+rf"$\mathrm{{d}} t$ = {dt*100}% of Mimas' period - with $J_2$")
        # plt.plot(times,Ekin)
        # plt.plot(times,Epot)
        plt.plot(times,Etot)
        plt.xlabel('Time (days)')
        plt.ylabel('Total energy (Rebound units)')

        plt.show()
        # plt.savefig(f"testdt/6SATdt{int(dt*100)}-J2.png", format='png')

##PLOTS YOU WANT TO SEE:
plot_n(False)
plot_Omega(False)
plot_OmegaDot(False)
plot_varpi(False)

plot_orbs(False)
plot_ecc(False)
plot_xt(False)

plot_resonant(False)
plot_dres_dt(True)
# plot_gradresangle(False)

plot_Energ(False)