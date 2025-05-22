import numpy as np
from numpy.random import poisson


def system(pop, t, cg):
    """
    Odes of dynamical system of  symbiont and host

    Args
    ====
    pop (np.array): population density of free-living symbiont (F), association (A), and host (H)
    t : argument to run the function odeint to iterate the ode
    cg (Config): parameters

    Return
    ======
    [dFdt, dAdt, dHdt]
    """
    F, A, H = pop
    dFdt = cg.rho * F + cg.tau * A - cg.alpha * F * F - cg.mu * F - cg.beta * F * H
    dAdt = (
        cg.r * cg.p * A + cg.beta * F * H - cg.gamma * (A + H) * A - (cg.d + cg.nu) * A
    )
    dHdt = (
        cg.r * H
        + cg.r * (1 - cg.p) * A
        - cg.gamma * (A + H) * H
        - cg.d * H
        - cg.beta * F * H
    )
    return [dFdt, dAdt, dHdt]


def sys_birth(pop, cg, V):
    """
    birth rate of the freeliving symbiont, association, and host

    Args
    ====
    pop (np.array): number of freeliving symbtion, association and host
    cg (Config): parameters
    V (int): volumn for population density

    Return
    ======
    np.array([birth rate F, birth rate A, birth rate H])
    """
    F, A, H = pop / V
    bF = cg.rho * F + cg.tau * A
    bA = cg.r * cg.p * A + cg.beta * F * H
    bH = cg.r * H + cg.r * (1 - cg.p) * A
    return np.array([bF, bA, bH])


def sys_death(pop, cg, V):
    """
    death rate of the freeliving symbiont, association, and host

    Args
    ====
    pop (np.array): number of freeliving symbtion, association and host
    cg (Config): parameters
    V (int): volumn for population density

    Return
    ======
    np.array([death rate F, death rate A, death rate H])
    """
    F, A, H = pop / V
    dF = cg.alpha * F * F + cg.mu * F + cg.beta * F * H
    dA = cg.gamma * (A + H) * A + (cg.d + cg.nu) * A
    dH = cg.gamma * (A + H) * H + cg.d * H + cg.beta * F * H
    return np.array([dF, dA, dH])


def tau_leap(init, birth_rate, death_rate, Tmax, dt, cg, V):
    """
    tau leap algorithm for stochastic dynamics of a system without mutation

    Args
    ====
    init (np.array): initial population numbers
    birth_rate (np.array): birth rate of populations
    death_rate (np.array): death rate of populations
    Tmax (int): maximum time for iteration
    dt (float): tau values for the leap
    cg (Config): parameters
    V (int): volumn to calculate the population density
    """
    state = [init]
    time = [0]
    tseries = np.arange(0, Tmax - 1, 1)
    for i in tseries:
        br = birth_rate(init, cg, V)
        bn = poisson(br * dt)
        dr = death_rate(init, cg, V)
        dn = poisson(dr * dt)
        init = init + bn - dn
        if any(init > 0):
            init = init * (init > 0)
            state.append(init)
            time.append(i + 1)
        else:
            break
    return dict(t=time, x=np.array(state))
