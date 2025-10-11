import rebound

sim = rebound.Simulation()
sim.units = ('AU', 'days', 'Msun')

# We can add Saturn and its moons by name, since REBOUND is linked to the HORIZONS database.
labels = ["699","Mimas","Tethys","Titan", "Enceladus", "Dione", "Rhea"]
# labels = ["699","Mimas","Tethys","Titan"]
sim.add(labels)

sim.save_to_file("DATA.bin")