import numpy as np
from numpy.random import poisson


def tau(rho, nu, cg):
    assert isinstance(rho, np.ndarray), "mutant rho has to be a numpy array"
    assert isinstance(nu, np.ndarray), "mutant nu value has to be a numpy array"
    return cg.theta - cg.v * rho**cg.h + (cg.eta * nu * (nu + cg.d) ** cg.g) / cg.nu_max


def F_birth_mut(pop_F, pop_A, cg, V, rho_mut, nu_mut):
    assert isinstance(rho_mut, np.ndarray), "mutant rho has to be a numpy array"
    assert isinstance(nu_mut, np.ndarray), "mutant nu value has to be a numpy array"
    assert isinstance(pop_F, np.ndarray), "F population has to be a numpy array"
    assert isinstance(pop_A, np.ndarray), "A population has to be a numpy array"
    return rho_mut * pop_F / V + tau(rho_mut, nu_mut, cg) * pop_A / V


def A_birth_mut(pop_F, pop_A, pop_H, cg, V):
    assert isinstance(pop_F, np.ndarray), "F population has to be a numpy array"
    assert isinstance(pop_A, np.ndarray), "A population has to be a numpy array"
    return cg.r * cg.p * pop_A / V + cg.beta * (pop_F / V) * (pop_H / V)


def H_birth_mut(pop_A, pop_H, cg, V):
    assert isinstance(pop_A, np.ndarray), "A population has to be a numpy array"
    assert len(pop_H) == 1, "There is always one host population"
    return cg.r * pop_H / V + cg.r * (1 - cg.p) * pop_A.sum() / V


def F_death_mut(pop_F, pop_H, cg, V):
    assert isinstance(pop_F, np.ndarray), "F population has to be a numpy array"
    return (
        cg.alpha * (pop_F / V) * (pop_F.sum()) / V
        + cg.mu * pop_F / V
        + cg.beta * (pop_F / V) * (pop_H / V)
    )


def A_death_mut(pop_F, pop_A, pop_H, cg, V, nu_mut):
    assert isinstance(nu_mut, np.ndarray), "mutant nu value has to be a numpy array"
    assert isinstance(pop_F, np.ndarray), "F population has to be a numpy array"
    assert isinstance(pop_A, np.ndarray), "A population has to be a numpy array"
    return (
        cg.gamma * ((pop_A.sum() + pop_H) / V) * pop_A / V + (cg.d + nu_mut) * pop_A / V
    )


def H_death_mut(pop_F, pop_A, pop_H, cg, V):
    assert isinstance(pop_F, np.ndarray), "F population has to be a numpy array"
    assert isinstance(pop_A, np.ndarray), "A population has to be a numpy array"
    assert len(pop_H) == 1, "There is always one host population"
    return (
        cg.gamma * ((pop_A.sum() + pop_H) / V) * pop_H / V
        + cg.d * pop_H / V
        + cg.beta * (pop_F.sum() / V) * (pop_H / V)
    )


def tau_leap_symbiosis_mutation(
    pop_init, Tmax, dt, cg, V, trait_init, sigma_rho, sigma_nu, mutant_rate, step
):
    F_state, A_state, H_state = [], [], []
    rho_val, nu_val, tau_val = [], [], []
    time = []
    ttfull = np.arange(0, Tmax, step)
    Fs, As, Hs = pop_init[0:1], pop_init[1:2], pop_init[2:3]
    rhos, nus, taus = (
        trait_init[0:1],
        trait_init[1:2],
        tau(trait_init[0:1], trait_init[1:2], cg),
    )
    for i in range(Tmax):
        if (any(Fs > 0) or any(As > 0)) and (Hs > 0):
            # if any population number is negative, set it to zero
            Fs = np.where(Fs > 0, Fs, 0)
            As = np.where(As > 0, As, 0)
            Hs = np.where(Hs > 0, Hs, 0)
            br = A_birth_mut(Fs, As, Hs, cg, V)  # rate
            dr = A_death_mut(Fs, As, Hs, cg, V, nus)  # rate
            A_change = poisson(br * dt) - poisson(dr * dt)  # intergers
            br = H_birth_mut(As, Hs, cg, V)  # rate
            dr = H_death_mut(Fs, As, Hs, cg, V)  # rate
            H_change = poisson(br * dt) - poisson(dr * dt)  # intergers
            br = F_birth_mut(Fs, As, cg, V, rhos, nus)  # rate
            dr = F_death_mut(Fs, Hs, cg, V)  # rate
            bn = poisson(br * dt)
            dn = poisson(dr * dt)
            F_change = bn - dn  # intergers
            # Number of mutants arise with each parent
            new_mut_arise = np.random.binomial(F_change * (F_change > 0), mutant_rate)
            ismutate = new_mut_arise > 0
            # arised mutants with corresponding parents' value
            rh = (rhos + np.random.normal(0, sigma_rho)) * ismutate
            nn = (nus + np.random.normal(0, sigma_nu)) * ismutate
            tt = tau(rh, nn, cg)
            # Only keep the trait values if they are within range
            nn_cond = (-cg.d <= nn) * (nn <= cg.nu_max)
            tt_cond = tt >= 0
            mut_cond = (
                nn_cond * tt_cond
            )  # final conditions are the combination of conditions for both nu and rho
            rh = rh * mut_cond
            nn = nn * mut_cond
            tt = tt * mut_cond
            # Even if new mutants arise, if the traits is not within range, it does not count
            new_mut_arise = new_mut_arise * mut_cond
            # Change in F population after mutation
            F_change_after_mut = F_change - new_mut_arise
            if any(ismutate * mut_cond):
                new_mut_nb = new_mut_arise[new_mut_arise > 0]
                new_rho = rh[rh != 0]
                new_nu = nn[nn != 0]
                new_tau = tau(new_rho, new_nu, cg)
                A_new = np.zeros(len(new_mut_nb))
            else:
                new_mut_nb, A_new = (), ()
                new_rho, new_nu, new_tau = (), (), ()
            # Update values on next time step
            As = np.concatenate((As + A_change, A_new), axis=None)
            Hs = Hs + H_change
            Fs = np.concatenate((Fs + F_change_after_mut, new_mut_nb), axis=None)
            rhos = np.concatenate((rhos, new_rho), axis=None)
            nus = np.concatenate((nus, new_nu), axis=None)
            taus = np.concatenate((taus, new_tau), axis=None)
            # Save values with manual step
            if i in ttfull:
                A_state.append(As)
                H_state.append(Hs)
                F_state.append(Fs)
                rho_val.append(rhos)
                nu_val.append(nus)
                tau_val.append(taus)
                time.append(i)
        else:
            print(
                f"Simulation stop at time step {i} as total Fs = {Fs.sum()}, total As = {As.sum()}, Hs = {Hs.sum()}"
            )
            break
    return dict(
        t=time,
        x_F=F_state,
        x_A=A_state,
        x_H=H_state,
        rho=rho_val,
        nu=nu_val,
        tau=tau_val,
    )
