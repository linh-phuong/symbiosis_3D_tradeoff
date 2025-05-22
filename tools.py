import numpy as np
from numpy.random import poisson
import copy as cp
import pandas as pd


class Config:
    def __init__(self, params):
        params = cp.deepcopy(params)
        self.__dict__["_params"] = params
        for k, v in params.items():
            self.__dict__[k] = v

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __setattr__(self, key, value):
        self._params[key] = value
        self.__dict__[key] = value

    def clone_me(self):
        return Config(self._params)

    def update(self, kwargs):
        return Config({**self._params, **kwargs})

    def _repr_html_(self):
        return pd.Series(self._params).to_frame().T._repr_html_()

    def getkeys(self):
        return list(self._params.keys())


def sort_array_by_reference(data_array, reference_array, axis=0):
    """
    Sort columns of a 2D array based on a reference array.

    Parameters:
    -----------
    data_array : numpy.ndarray
        The 2D array to be sorted (shape should be (10, 3))
    reference_array : numpy.ndarray
        The array to use for sorting (used to determine sort indices)
    axis : int, optional
        Axis along which to sort (default is 0, sorting columns)

    Returns:
    --------
    numpy.ndarray
        Sorted array with columns rearranged based on reference_array
    """
    # Get the sorting indices based on the reference array
    sort_indices = np.argsort(reference_array)

    # Use advanced indexing to rearrange the columns
    if axis == 0:
        sorted_array = data_array[:, sort_indices]
    elif axis == 1:
        sorted_array = data_array[sort_indices, :]
    else:
        raise ValueError("axis must be 0 or 1")

    return sorted_array


def turn_raw_dat_to_sorted_arrays(dat):
    xF_arr, xA_arr = [], []
    r_arr, n_arr, t_arr = [], [], []
    x_F, x_A = dat["x_F"], dat["x_A"]
    rho, nu, tau = dat["rho"], dat["nu"], dat["tau"]
    assert len(x_F[-1]) == len(x_A[-1]) == len(rho[-1]) == len(nu[-1]) == len(tau[-1])
    tt = len(x_F[-1])
    tstep = len(dat["t"])
    for i in range(tstep):
        assert len(x_F[i]) == len(x_A[i]) == len(rho[i]) == len(nu[i]) == len(tau[i])
        cc = tt - len(x_F[i])
        xx = np.concatenate((x_F[i], np.zeros(cc)))
        xF_arr.append(xx)
        xx = np.concatenate((x_A[i], np.zeros(cc)))
        xA_arr.append(xx)
        xx = np.concatenate((rho[i], np.zeros(cc)))
        r_arr.append(xx)
        xx = np.concatenate((nu[i], np.zeros(cc)))
        n_arr.append(xx)
        xx = np.concatenate((tau[i], np.zeros(cc)))
    xA_rho = sort_array_by_reference(np.array(xA_arr), rho[-1])
    xA_nu = sort_array_by_reference(np.array(xA_arr), nu[-1])
    xA_tau = sort_array_by_reference(np.array(xA_arr), tau[-1])
    xF_rho = sort_array_by_reference(np.array(xF_arr), rho[-1])
    xF_nu = sort_array_by_reference(np.array(xF_arr), nu[-1])
    xF_tau = sort_array_by_reference(np.array(xF_arr), tau[-1])
    return dict(
        xA_rho=xA_rho,
        xA_nu=xA_nu,
        xA_tau=xA_tau,
        xF_rho=xF_rho,
        xF_nu=xF_nu,
        xF_tau=xF_tau,
        rho_sort=np.sort(rho[-1]),
        nu_sort=np.sort(nu[-1]),
        tau_sort=np.sort(tau[-1]),
    )
