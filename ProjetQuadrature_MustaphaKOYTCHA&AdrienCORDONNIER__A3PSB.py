"""
----------------Etude du système de Brusselator à l'aide d'EDO----------------

Ce programme permet de calculer des systèmes d'équations différentielles à l'aide des méthodes suivantes :
- Euler_explicit(f, ic, T0, Tf, N)
- RKF4Butcher(f, ic, T0, Tf, N)
- RK4(f, ic, T0, Tf, N)
- stepRK45(f, ic, T0, Tf, epsilonmax)

Chacune de ses fonction prend en paramètre :
* f : une fonction
* ic : un vecteur contetant les conditions initiales du système à étudier
* T0 & Tf : le temps initial et final de l'étude
* N : le nombre de subdivision pour la discrétisation de notre problème
* epsilonmax : une tolérance souhaitée

On trace chacune des solutions à l'aide de :
- concentration_plotting(t, y, NomMethode, a, b)
- trajectory_plotting(y, NomMethode, a, b)

L'étude porte sur la fonction de Brusselator avec les paramètres (a, b) =
* (1, 1.5)
* (1, 3)

Auteurs : Adrien CORDONNIER & Mustapha KOYTCHA
Date : 18/04/2023
"""

#Bibiothèques
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint

# Fonction du système de Brusselator
def Brusselator(a, b):
    def f(Y):
        x, y = Y
        dxdt = a + x**2 * y - (b+1)* x
        dydt = b*x - x**2*y
        return np.array([dxdt, dydt])
    return f

# Schéma d'EULER explicite
def Euler_explicit(f, ic, T0, Tf, N):
    t0 = time.time()
    h = (Tf - T0) / N
    Lt = np.arange(T0, Tf + h, h)
    Ly = np.zeros((len(Lt), len(ic)))
    Ly[0] = ic

    for n in range(len(Lt) - 1):
        Ly[n+1] = Ly[n] + h * f(Ly[n])

    t1 = time.time()
    print("Temps méthode EULER explicite : ", t1-t0, "s")
    return Lt, Ly


# Fonction pour afficher les graphiques des concentrations x et y en fonction du temps
def concentration_plotting(t, y, NomMethode, a, b):
    plt.plot(t, y[:, 0], label="x")
    plt.plot(t, y[:, 1], label="y")
    plt.plot(t, [a]*len(t), "--", label="Asymptote à x")
    plt.plot(t, [b]*len(t), "--", label="Asymptote à y")
    plt.xlabel("Temps")
    plt.ylabel("Concentration")
    plt.title("Méthode : {t} \n Evolution temporelle des concentrations x et y (a={a} et b={b})".format(t=NomMethode,
                                                                                                        a=a,
                                                                                                        b=b))
    plt.savefig("Evol-concentration.pdf")
    plt.legend()
    plt.grid()
    plt.show()


# Fonction pour afficher le graphique de la trajectoire (x,y) en fonction du temps
def trajectory_plotting(y, NomMethode, a, b):
    plt.plot(y[:, 0], y[:, 1])
    plt.xlabel("Concentration x")
    plt.ylabel("Concentration y")
    plt.plot(a, b/a, "ro", label="(a, b/a) = " + str((a,b/a)))
    plt.title("Méthode : {t} \n Trajectoire de x et y (a={a} et b={b})".format(t=NomMethode,
                                                                                                        a=a,
                                                                                                        b=b))
    plt.savefig("trajectoire.pdf")
    plt.legend()
    plt.grid()
    plt.show()

# Schéma de RUNGE-KUTTA F4
def RKF4Butcher(f, ic, T0, Tf, N):
    t0 = time.time()
    h = (Tf - T0) / N  # step size if h is constant
    Lt = np.arange(T0, Tf + h, h)
    Ly = np.empty((len(Lt), np.size(ic)), dtype=float)  # Matrice contenant les valeurs de f(Y)
    Ly[0, :] = ic
    M = 6

    # Coefficients de Butcher
    beta = np.array([[0,           0,              0,              0,              0,              0],
                  [1/4,         0,              0,              0,              0,              0],
                  [3/32,        9/32,           0,              0,              0,              0],
                  [1932/2197,  -7200/2197,      7296/2197,      0,              0,              0],
                  [439/216,    -8,              3680/513,      -845/4104,       0,              0],
                  [-8/27,       2,             -3544/2565,      1859/4104,     -11/40,          0]])

    gamma = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    alpha = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])  # Pas utile ici

    K = [0] * M  # Liste contenant des Kij

    # Boucle principale
    for i in range(len(Lt) - 1):
        for k_j in range(M):
            if k_j == 0:
                K[k_j] = f(Ly[i,:])
            else:
                s = sum([beta[k_j, var] * K[var] for var in range(k_j)])
                K[k_j] = f(Ly[i,:] + h * s)
        Ly[i+1, :] = Ly[i, :] + h * sum([gamma[k_j] * K[k_j] for k_j in range(M)])

    t1 = time.time()
    print("Temps méthode RKF4Butcher : ", t1-t0, "s")
    return Lt, Ly


#Schéma de RUNGE-KUTTA 4
def RK4(f, ic, T0, Tf, N):
      t0 = time.time()
      h = (Tf - T0) / N      #step size if h is constant
      Lt = np.linspace(T0, Tf, N)
      Ly = np.empty((N, np.size(ic)),dtype = float)
      Ly[0,:] = ic
      for i in range(N-1):
          #if h isn't constant, we use h=t[i+1]-t[i]
          k1 = h*f(Ly[i,:])
          y1 = Ly[i,:] + 1/2*k1
          k2 = h* f(y1)
          y2 = Ly[i,:] + 1/2*k2
          k3 = h* f(y2)
          y3 = Ly[i,:] + k3
          k4 = h* f(y3)
          k = (k1+2*k2+2*k3+k4)/6
          Ly[i+1,:] = Ly[i,:] + k
      t1 = time.time()
      print("Temps méthode RUNGE-KUTTA 4 : ", t1 - t0, "s")
      return Lt, Ly

# Schéma de RUNGE-KUTTA F4 à pas adaptatif
def stepRK45(f, ic, T0, Tf, epsilonmax=10**-4):
    t0 = time.time()
    h = 10  # Pas initial peu important
    Lt = [T0]
    Ly = np.zeros((1, 2))  # Matrice contenant les valeurs de f(Y)
    Ly[0, :] = ic
    M = 6

    # Coefficients de Butcher
    beta = np.array([[0, 0, 0, 0, 0, 0],
                     [1 / 4, 0, 0, 0, 0, 0],
                     [3 / 32, 9 / 32, 0, 0, 0, 0],
                     [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
                     [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
                     [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0]])

    gamma = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
    gamma_bar = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
    delta = gamma - gamma_bar

    alpha = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])  # Pas utile ici

    K = [0] * M  # Liste contenant des Kij
    i = 0
    l_h = []

    while Lt[-1] < Tf:
        for k_j in range(M):
            if k_j == 0:
                K[k_j] = f(Ly[i, :])
            else:
                s = sum([beta[k_j, var] * K[var] for var in range(k_j)])
                K[k_j] = f(Ly[i, :] + h * s)

        E = h * sum([delta[k_j] * K[k_j] for k_j in range(M)])  # Calcul de l'erreur
        epsilon = np.linalg.norm(E, np.inf)  # Calcul de la norme infinie de l'erreur

        if epsilon < epsilonmax:
            Lt.append(Lt[-1] + h)
            Ly = np.vstack([Ly, Ly[i, :] + h * sum([gamma[k_j] * K[k_j] for k_j in range(M)])])
            l_h.append(h)
            i += 1

        e = 0.9 * (epsilonmax / epsilon) ** (1 / 5)

        if e < 0.1:
            h *= 0.1
        elif e > 5:
            h *= 5
        else:
            h *= e

    t1 = time.time()
    print("Temps méthode RUNGE-KUTTA 4 Pas adaptatif : ", t1-t0, "s")
    return Lt, Ly, l_h


# Paramètres de l'étude
sys = [(1, 1.5), (1, 3)]
ic = np.array([0, 1])
t0, t1 = 0, 18
N = 1000

for i in sys:
    a, b = i
    f = Brusselator(a, b)

    print("--------------Pour a={x}, b={y}--------------".format(x=a, y=b), "\n")

    # Exécutions
    t, y = Euler_explicit(f, ic, t0, t1, N)
    concentration_plotting(t, y, "EULER explicite", a, b)
    trajectory_plotting(y, "EULER explicite", a, b)

    t, y = RKF4Butcher(f, ic, t0, t1, N)
    concentration_plotting(t, y, "RUNGE-KUTTA F 4", a, b)
    trajectory_plotting(y, "RUNGE-KUTTA F 4", a, b)

    t, y = RK4(f, ic, t0, t1, N)
    concentration_plotting(t, y, "RUNGE-KUTTA 4", a, b)
    trajectory_plotting(y, "RUNGE-KUTTA 4", a, b)

    t, y, h = stepRK45(f, ic, t0, t1)
    concentration_plotting(t, y, "RUNGE-KUTTA 4 Butcher Pas adaptatif", a, b)
    trajectory_plotting(y, "RUNGE-KUTTA 4 Butcher Pas adaptatif", a, b)

    plt.figure()
    plt.plot(t[:-1], h)
    plt.title("Variations du pas de temps en fonction du temps")
    plt.xlabel("Temps")
    plt.ylabel("h")
    plt.grid()
    plt.show()

    print("\n")

# Fonction pour calculer l'erreur relative maximale entre deux solutions
def erreur_relative(y1, y2):
    return np.max(np.abs(y1 - y2) / np.maximum(1e-10, np.abs(y2)))

a, b = 1, 1.5
f = Brusselator(a, b)
ic = np.array([0, 1])
T0, Tf = 0, 18
N = 1000
epsilonmax = 1e-6

# Euler_explicit vs RKF4Butcher
Lt_euler, Ly_euler = Euler_explicit(f, ic, T0, Tf, N)
Lt_rkf4, Ly_rkf4 = RKF4Butcher(f, ic, T0, Tf, N)
erreur_rel_euler_rkf4 = np.abs(Ly_euler - Ly_rkf4) / np.maximum(1e-10, np.abs(Ly_rkf4))
t = np.linspace(0,18,1002)
plt.plot(t, erreur_rel_euler_rkf4, label='Euler_explicit vs RKF4Butcher')
plt.title("Variations de l'erreur relative en fonction du temps")
plt.legend()
plt.show()

# Euler_explicit vs RK4
Lt_euler, Ly_euler = Euler_explicit(f, ic, T0, Tf, N)
Lt_rk4, Ly_rk4 = RK4(f, ic, T0, Tf, N)
Ly_euler = np.delete(Ly_euler, -1, axis=0)
Ly_euler = np.delete(Ly_euler, -1, axis=0)
erreur_rel_euler_rk4 = np.abs(Ly_euler - Ly_rk4) / np.maximum(1e-10, np.abs(Ly_rk4))
t = np.linspace(0,18,1000)
plt.plot(t, erreur_rel_euler_rk4, label='Euler_explicit vs RK4')
plt.title("Variations de l'erreur relative en fonction du temps")
plt.legend()
plt.show()

# RKF4Butcher vs RK4
Ly_rkf4, Ly_rkf4 = RKF4Butcher(f, ic, T0, Tf, N)
Lt_rk4, Ly_rk4 = RK4(f, ic, T0, Tf, N)
Ly_rkf4 = np.delete(Ly_rkf4, -1, axis=0)
Ly_rkf4 = np.delete(Ly_rkf4, -1, axis=0)
erreur_rel_rkf4_rk4 = np.abs(Ly_rkf4 - Ly_rk4) / np.maximum(1e-10, np.abs(Ly_rk4))
plt.plot(t, erreur_rel_rkf4_rk4, label='RKF4Butcher vs RK4')
plt.title("Variations de l'erreur relative en fonction du temps")
plt.legend()
plt.show()