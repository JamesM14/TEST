import numpy as np
from scipy.integrate import odeint

# Define the system of differential equations
def model(y, t, r_0, r_a, K_0, K_e, P, S_0, S_a, a_0, a_e, e):
    N, a, S = y
    r = r_0 + r_a * a
    K = K_0 + K_e * e
    a = a_0 + a_e * e
    S = S_0 + S_a * a
    dNdt = r * N * (1 - N / K) - P * N
    da_dt = a_0 + a_e * e
    dS_dt = S_0 + S_a * a
    return [dNdt, da_dt, dS_dt]

# Check if the system is at equilibrium
def is_at_equilibrium(y, t, r_0, r_a, K_0, K_e, P, S_0, S_a, a_0, a_e, e):
    dNdt, da_dt, dS_dt = model(y, t, r_0, r_a, K_0, K_e, P, S_0, S_a, a_0, a_e, e)
    return abs(dNdt) < 1e-6 and abs(da_dt) < 1e-6 and abs(dS_dt) < 1e-6

# Initial conditions
N0 = 1000  # initial population
a0 = 1     # initial resource availability
S0_values = np.linspace(0.1, 0.9, 5)  # initial sex ratios
P_values = np.linspace(0.01, 0.1, 5)  # predation rates

# Parameters
r_0 = 1
r_a = 0.1
K_0 = 5000
K_e = 0.1
S_0 = 0.5
S_a = 0.1
a_0 = 1
a_e = 0.1
e = 1

# Run simulations for each combination of initial conditions
for S0 in S0_values:
    for P in P_values:
        # Set the initial conditions
        y0 = [N0, a0, S0]

        # Run the simulation until the system reaches equilibrium
        t = 0
        dt = 0.01
        y = y0
        while not is_at_equilibrium(y, t, r_0, r_a, K_0, K_e, P, S_0, S_a, a_0, a_e, e):
            t += dt
            y = odeint(model, y, [t, t+dt], args=(r_0, r_a, K_0, K_e, P, S_0, S_a, a_0, a_e, e))[-1]

        print(f"Initial conditions: S0={S0}, P={P}")
        print(f"Equilibrium reached at t={t}, N={y[0]}, a={y[1]}, S={y[2]}")
