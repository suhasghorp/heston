# distutils: language = c++
"""
@author: Suhas Ghorp
 python setup.py build_ext --inplace
 python main.py
 cython HestonPutCombined.pyx -a => view html in browser
"""
import numpy as np
import cython
from libc.math cimport sqrt, fabs, fmax, pow, exp, erfc, log

cdef inline double v_next_euler(double v_minus, double kappa, double theta,double dt,double sigma,double z) nogil except +:
    return v_minus + kappa * (theta - v_minus) * dt + sigma * sqrt(v_minus) * sqrt(dt) * z
cdef inline double v_next_milstein(double v_minus, double kappa, double theta,double dt,double sigma,double z) nogil except +:
    return v_minus + kappa * (theta - v_minus) * dt + sigma * sqrt(
            v_minus) * sqrt(dt) * z + 1 / 4 * pow(sigma, 2) * dt * (pow(z, 2) - 1)
cdef inline double s_next(double s_minus,double r,double q, double v_minus,double dt,double Zs) nogil except +:
    return s_minus * exp((r - q - v_minus / 2) * dt + sqrt(v_minus) * sqrt(dt) * Zs)

cdef inline double normalCDF(double x) nogil except +:
    return erfc(-x / sqrt(2))/2.0

cpdef inline (double,double) BS_Put_Vega(double S, double E, double r, double v, double tau, double T) nogil except +:
    cdef:
        double d1,d2,N1,N2,C,P,vega

    if tau > 0:

        d1 = log(S / E) / v * float(sqrt(T)) + (r + (v * v) / 2) * T / (v * sqrt(T))
        d2 = d1 - float(v * (T ** 0.5))
        N1 = normalCDF(d1)
        N2 = normalCDF(d2)
        C = S * N1 - E * exp(-r * (tau)) * N2
        P = C + E * exp(-r * tau) - S
        vega = S * sqrt(T) * normalCDF(d1)

    else:
        P = max(E - S, 0)
        vega = 0
    return P, vega


@cython.boundscheck(False)
@cython.wraparound(False)
def EulerMilsteinSim(scheme, negvar, long numPaths, double rho, double S_0, double V_0, double T, double kappa,
                     double theta, double sigma, double r, double q):

    cdef double dt = 0.001
    cdef int num_time = int(T / dt)
    S_array = np.zeros((num_time + 1, numPaths), dtype=np.double)
    cdef double[:,:] S = S_array

    S[0, :] = S_0
    V_array = np.zeros((num_time + 1, numPaths),dtype=np.double)
    cdef double[:,:] V = V_array
    V[0, :] = V_0
    cdef int Vcount0 = 0
    cdef long i, t_step
    cdef double Zv, Zs
    rng = np.random.default_rng()
    cdef double[:,:] Z = rng.standard_normal(numPaths * (2 * num_time)).reshape((numPaths,2*num_time))
    with nogil:
        for i in range(numPaths):
            for t_step in range(1, num_time + 1):
                Zv = Z[i,t_step]
                Zs = rho * Zv + sqrt(1 - pow(rho, 2)) * Z[i,t_step+1]

                if scheme == 'Euler':
                    V[t_step,i] = v_next_euler(V[t_step - 1, i],kappa,theta,dt,sigma,Zv)
                elif scheme == 'Milstein':
                    V[t_step, i] = v_next_milstein(V[t_step - 1, i], kappa, theta, dt, sigma, Zv)

                if V[t_step, i] <= 0:
                    Vcount0 = Vcount0 + 1
                    if negvar == 'Reflect':
                        V[t_step, i] = fabs(V[t_step, i])
                    elif negvar == 'Trunca':
                        V[t_step, i] = fmax(V[t_step, i], 0)

                S[t_step, i] = s_next(S[t_step - 1, i],r,q,V[t_step - 1, i],dt,Zs)

    return S, V, Vcount0

@cython.boundscheck(False)
@cython.wraparound(False)
def EulerMilsteinPrice(scheme, negvar, long numPaths, double rho, double S_0, double V_0, double Tmax,
                       double kappa, double theta, double sigma, double r, double q, double[:] maturities,
                       double[:] strikes):
    cdef:
        int num_maturities = len(maturities)
        int num_strikes = len(strikes)
        double dt = 0.001
        double[:,:] OptionPriceMatrix = np.zeros((num_strikes, num_maturities),dtype=np.double)
        double[:,:]  stdErrTable = np.zeros((num_strikes, num_maturities),dtype=np.double)
        int i,j,T_row
        double T_temp, KK,stdDev,stdErr,SimPrice
        double[:] S_T,Payoff


    S, V, Vcount0 = EulerMilsteinSim(scheme, negvar, numPaths, rho, S_0, V_0, Tmax, kappa, theta, sigma, r, q)

    for i in range(num_maturities):
        T_temp = maturities[i]
        T_row = int(T_temp / dt)
        S_T = S[T_row, :]

        for j in range(num_strikes):
            KK = strikes[j]
            Payoff = np.asarray([fmax(KK - x, 0) for x in S_T])
            SimPrice = np.exp(-r * T_temp) * np.mean(Payoff)
            OptionPriceMatrix[j][i] = SimPrice
            stdDev = np.std(Payoff,dtype=np.double)
            stdErr = stdDev / sqrt(numPaths)
            stdErrTable[j][i] = stdErr
    return S, V, Vcount0, OptionPriceMatrix, stdErrTable, Payoff