from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from math import exp, log, sqrt, pi
import numpy as np
from HestonPutCombined import EulerMilsteinPrice,BS_Put_Vega # import the Hestonston simulation
from scipy.stats import norm

import time


def impVolBsPut(S, E, r, T, P_heston):
    """
    This is to find the implied vol of a Black Scholes put option given ...
    the asset price S, exercise price E, interest rate r, maturity T, the Heston Option Price P_heston
    """



    """
    Newton Raphson Algorithm 
    """
    tau = T
    tol = 10e-15
    sigma = 0.2  # initial guess of the implied volatility
    sigmadiff = 1.
    k = 1  # index for search
    kmax = 100  # max number of searches

    while sigmadiff > tol and k < kmax:
        (P, Pvega) = BS_Put_Vega(float(S), float(E), float(r), float(sigma), float(tau), float(T))
        if Pvega == 0.0:
            #print("Pvega = 0.0")
            Pvega = float(0.01)
            # return 0
            # Pvega = 0.2

        increment = (P - float(P_heston)) / Pvega
        sigma = float(sigma) - increment
        sigmadiff = abs(increment)  # stop searching when the increment is smaller than the tolerance
        k = k + 1
    return sigma


if __name__ == "__main__":
    start_time = time.time()

    strikes = np.arange(50, 100, 0.1)
    maturities = np.arange(0.5, 3 + 0.1, 0.1)

    S_0 = 75
    r = 0.02

    (S, V, Vcount0, OptionPriceMatrix, stdErrTable, Payoff) = \
        EulerMilsteinPrice('Milstein', 'Trunca', numPaths=500, rho=-0.6, S_0=S_0, V_0=(0.1) ** 2, \
                           Tmax=3, kappa=0.5, theta=(0.25) ** 2, sigma=0.1, r=r, q=0.0, maturities=maturities, \
                           strikes=strikes)

    print("--- %s seconds ---" % (time.time() - start_time))

    ImpVolTable = np.zeros((len(strikes), len(maturities)))

    for i in range(len(maturities)):
        maturity = maturities[i]
        for j in range(len(strikes)):
            K = strikes[j]
            price_heston = OptionPriceMatrix[j][i]
            ImpVolTable[j][i] = impVolBsPut(S_0, K, r, maturity, price_heston)
    print("done")

    plt.close()

    fig = plt.figure(1)
    plt.plot(S)
    plt.ylabel('Asset Price S from Heston model')
    plt.xlabel('time step')

    fig = plt.figure(2)
    volatility = np.sqrt(V)
    plt.plot(volatility)
    plt.ylabel('Stochastic volatility V from Heston model')
    plt.xlabel('time step')

    fig = plt.figure(3)
    ax = plt.axes(projection='3d')
    xx, yy = np.meshgrid(strikes, maturities)
    ax.plot_surface(xx, yy, OptionPriceMatrix.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Exercise Price K')
    ax.set_ylabel('Maturity T ')
    ax.set_zlabel('Option price ')

    fig = plt.figure(4)
    ax = plt.axes(projection='3d')
    xx, yy = np.meshgrid(strikes, maturities)
    ax.plot_surface(xx, yy, ImpVolTable.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Exercise Price K')
    ax.set_ylabel('Maturity T ')
    ax.set_zlabel('Implied Volatility ')
    ax.set_zlim(0.1, 0.25)

    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))