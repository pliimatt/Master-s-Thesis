from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sys import argv

SIM_TIMES = np.linspace(0, 100, 10000) # dt = t[-1] / len(t)

#---------------------------------
# REMEMBER TO CHANGE THE NUMBER OF CPUS!!
#---------------------------------

CPUs = 30
ST = 100

N = 7
noStates = 3
qi = N // 2

W = 25
U = 3.5
#J = 1

if len(argv)>1:
    W = int(argv[1])

# Save the picture as saveName.png and data as saveName.data
saveName = f"sim"
plot_title = f"The population difference of the lowest 2 states"

print(f"""t = {SIM_TIMES[0]} - {SIM_TIMES[-1]}, dt = 1/{int(len(SIM_TIMES)/SIM_TIMES[-1])}
Realizations = {ST}
Number of qubits = {N}
W = {W}
U = {round(U, 3)}
transmon has {noStates} states
studying transmon #{qi + 1}
using {CPUs} CPUs
""")


#-----------------------------------------------------------------------------------------
def interval(j, n):
    """
    Optimal intervals for the pulses in UDD
    j: current index of pulse
    n: maximum number of pulses
    """
    return np.sin(np.pi * j / (2 * n + 2)) ** 2
#-----------------------------------------------------------------------------------------
def plot_data(x, y, name):
    """
    Plot data with name
    """
    fig, ax = plt.subplots()
    ax.plot(x, y, label = name)
    plt.legend(loc=1)
    plt.grid(which = "both", axis = "both")
    plt.title(plot_title)
    plt.savefig(saveName + f"{name}", dpi = 120)
    
    return
#-----------------------------------------------------------------------------------------
def fittingF(x, A, B, C, om, r):
    return  (A * np.cos(om * x) + B * np.sin(om * x) ) * np.exp(-r * x) + C
#-----------------------------------------------------------------------------------------
def analyze(data, tol=1e-8):
    """
    Analyze the fitting of fittingF to the sample data
    """
    if tol > 1:
        print(f"Too big tolerance!")
        return

    print(f"Starting the analysis with tolerance {tol}")
    y = data[10:]
    x = SIM_TIMES[10:]

    # Fit to x
    try:
        fit_params, covm = curve_fit(fittingF, x, y, ftol=tol, xtol=tol)
    except RuntimeError:
        print("Could not fit to the function")
        print(f"Trying again with increased tolerance")
        analyze(data, 10 * tol)
        return

    # Calculate the fitted curve
    fit = fittingF(x, *fit_params)
    
    print(f"Fitted: ( A cos omega t  + B sin omega t ) exp - R_2 t + C")
    print("Parameters with their errors:")
    print(f"A = {fit_params[0]}, {np.sqrt(covm[0][0])}")
    print(f"B = {fit_params[1]}, {np.sqrt(covm[1][1])}")
    print(f"C = {fit_params[2]}, {np.sqrt(covm[2][2])}")
    print(f"omega = {fit_params[3]}, {np.sqrt(covm[3][3])}")
    print(f"R_2 = {fit_params[4]}, {np.sqrt(covm[4][4])}")

    # Display the fitted curve
    fig, ax = plt.subplots()
    ax.plot(SIM_TIMES, data, label = "True curve")
    ax.plot(x, fit, label = "Fitted curve")
    plt.legend(loc=1)
    plt.grid(which = "both", axis = "both")
    plt.title(plot_title)
    plt.savefig(saveName + "Fitting", dpi = 120)
    
    return
#-----------------------------------------------------------------------------------------
def Ham1(E):
    """
    First term in BH Hamiltonian: sum of the energy operators
    """
    np.random.seed(None)
    
    H1 = []
    for a in range(N):
        # H1 = (n x I x ... x I) + (I x n x ... x I) + ...
        b = tensor([ qeye(noStates) for _ in range(a) ] + [num(noStates)] + [ qeye(noStates) for _ in range( N - a - 1 ) ])
        H1.append(b * E[a])
    
    return sum(H1)
#-----------------------------------------------------------------------------------------
def Ham2():
    """
    Second term in BH Hamiltonian: unharmonicity
    Useless if noStates < 3
    """
    H2 = []
    for a in range(N):
        # H2 = -U/2 * ( (n(n-1) x I x ... x I) + (I x n(n-1) x ... x I) + ... )
        b = tensor([ qeye(noStates) for _ in range(a) ] + [num(noStates) * (num(noStates) - qeye(noStates))] + [ qeye(noStates) for _ in range( N - a - 1 ) ])
        H2.append(b * -U / 2)
    
    return sum(H2)
#-----------------------------------------------------------------------------------------
def Ham3():
    """
    Third term in BH Hamiltonian: interaction (J = 1)
    """
    H3 = []
    for a in range(N - 1):
        # H3 = (a x a^ x I x ... x I) + (I x a x a^ x ... x I)
        b1 = tensor([ qeye(noStates) for _ in range(a) ] + [create(noStates)] + [destroy(noStates)] + [ qeye(noStates) for _ in range( N - a - 2 ) ])
        b2 = tensor([ qeye(noStates) for _ in range(a) ] + [destroy(noStates)] + [create(noStates)] + [ qeye(noStates) for _ in range( N - a - 2 ) ])
        H3.append(b1 + b2)
    
    return sum(H3)
#-----------------------------------------------------------------------------------------
def Hamiltonian(E):
    """
    Bose-Hubbard Hamiltonian
    set number of qubits from N
    """
    return Ham1(E) + Ham2() + Ham3()
#-----------------------------------------------------------------------------------------
def timedep(t, args):
    return np.cos(args["E"] * t)
#-----------------------------------------------------------------------------------------
def sim(run_no):
    """
    Simulate N qubits over time in SIM_TIMES
    return states observed by the probe
    """
    #generalized sigma-x operator
    sx = (create(noStates) + destroy(noStates))/2
    
    # density matrix of N qubits
    # at starting state | 0 1 0 1 ...>
    q = tensor([basis(noStates, a % 2) for a in range(N)])
    q = q.unit()
    
    # starting energies of N qubits
    E = [np.random.rand() * W for _ in range(N)]
    
    # BH Hamiltonian
    H0 = Hamiltonian(E)

    # Using the probe to get information
    probe = tensor([qeye(noStates) for _ in range( qi )] + [sx] + [qeye(noStates) for _ in range(N - qi - 1)])
    H = [H0, [probe, timedep]]
    
    res = sesolve(H, q, SIM_TIMES, args = {"E": E[ qi ]}).states
    
    rho = [a.ptrace(qi) for a in res]
    return [rho[t][1][0][1] - rho[t][0][0][0] for t in range(len(SIM_TIMES))]
#-----------------------------------------------------------------------------------------
def main():
    states = parallel_map(sim, range(ST), progress_bar = True, num_cpus = CPUs)
    #  Average sigma-z
    sz = np.real([np.average([s[t] for s in states]) for t in range(len(SIM_TIMES))])
    plot_data(SIM_TIMES, sz, "sigma-z")
    # Analyze the sigma-z graph
    #np.save(f"{saveName}-data.txt",sz)
    analyze(sz)
    
    return
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    print("Program ended successfully!")
#-----------------------------------------------------------------------------------------
