import numpy as np
from scipy.integrate import solve_ivp

# %% DESCRIPTION OF THE MODULE
"""
This file contains the Equations of the Tank of the rocket. These will
be integrated using the Runge-Kutta 4th order method.
"""


def TankRhoDer(t, y, rho_t, v_n, A_e, V):
    """
    input: V is the volume of the bottle
    """
    return -rho_t*v_n*A_e/V


def TankDensComp(rho_t, v_n, A_e, V, step):
    """
    Compute the density of the air inside the tank given initial
    conditions and the time step of integration.
    """
    rho_t = solve_ivp(TankRhoDer,
                      (0, step),
                      [rho_t],
                      args=(rho_t, v_n, A_e, V))
    return rho_t.y[0][-1]


def TankPressDer(t, y, P_atm, A_e, v_n, V, p, gamma=1.4):
    return 0.5*((gamma-1)*p - (gamma+1)*P_atm)*v_n*A_e/V


def TankPressComp(P_atm, A_e, v_n, V, p, step):
    """
    Compute the pressure of the air inside the tank given initial
    conditions and the time step of integration.
    """
    p = float(p)
    p = solve_ivp(TankPressDer, (0, step), [p], args=(P_atm, A_e, v_n, V, p))
    return p.y[0][-1]


def TempComp(p, rho, Rg=287):
    """
    Compute the Temperature from the ideal gas equation
    """
    return Rg*p/rho


def VnozzleComp(p, P_atm, rho_t):
    return np.sqrt((p - P_atm)/rho_t)
