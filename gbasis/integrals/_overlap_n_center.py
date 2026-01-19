"""N-center overlap integrals involving primitive Gaussians."""

import numpy as np


def _compute_gaussian_product_params(coords, exponents):
    r"""Compute the Gaussian product theorem parameters for N centers.

    For N primitive Gaussians centered at A_1, A_2, ..., A_N with exponents
    alpha_1, alpha_2, ..., alpha_N, the product is a single Gaussian with
    combined exponent gamma, combined center P, and pre-exponential factor K.

    Parameters
    ----------
    coords : np.ndarray(N, 3)
        Coordinates of the N Gaussian centers.
    exponents : np.ndarray(N,)
        Exponents of the N primitive Gaussians.

    Returns
    -------
    gamma : float
        Combined exponent.
    center_p : np.ndarray(3,)
        Combined center.
    factor_k : float
        Pre-exponential factor.

    """
    num_centers = len(exponents)

    # combined exponent
    gamma = np.sum(exponents)

    # combined center
    center_p = np.sum(exponents[:, np.newaxis] * coords, axis=0) / gamma

    # pre-exponential factor
    exponent_sum = 0.0
    for i in range(num_centers):
        for j in range(i + 1, num_centers):
            r_ij_squared = np.sum((coords[i] - coords[j]) ** 2)
            exponent_sum += exponents[i] * exponents[j] * r_ij_squared

    factor_k = np.exp(-exponent_sum / gamma)

    return gamma, center_p, factor_k


def _compute_primitive_s_overlap(coords, exponents):
    r"""Compute the overlap of N s-type (L=0) primitive Gaussians.

    Parameters
    ----------
    coords : np.ndarray(N, 3)
        Coordinates of the N Gaussian centers.
    exponents : np.ndarray(N,)
        Exponents of the N primitive Gaussians.

    Returns
    -------
    overlap : float
        The N-center s-type overlap integral.

    """
    gamma, _, factor_k = _compute_gaussian_product_params(coords, exponents)

    return (np.pi / gamma) ** 1.5 * factor_k


def _build_angular_momentum_recursion(
    coords, angmom_comps_list, gamma, center_p, factor_k
):
    r"""Build N-center overlap integrals for arbitrary angular momentum.

    Uses Obara-Saika recursion extended to N centers.

    Parameters
    ----------
    coords : np.ndarray(N, 3)
        Coordinates of the N Gaussian centers.
    angmom_comps_list : list of np.ndarray
        List of N arrays of angular momentum components.
    gamma : float
        Combined exponent.
    center_p : np.ndarray(3,)
        Combined center.
    factor_k : float
        Pre-exponential factor.

    Returns
    -------
    integrals : np.ndarray
        Array of overlap integrals.

    """
    n_centers = len(coords)
    output_shape = tuple(len(comps) for comps in angmom_comps_list)
    integrals = np.zeros(output_shape)

    # base case
    s_base = (np.pi / gamma) ** 1.5 * factor_k

    # memoization cache
    cache = {}

    def get_integral(angmom_tuple):
        """Recursively compute integral for given angular momentum tuple."""
        if angmom_tuple in cache:
            return cache[angmom_tuple]

        # base case: all angular momenta are zero
        if all(sum(a) == 0 for a in angmom_tuple):
            cache[angmom_tuple] = s_base
            return s_base

        # find first center with non-zero angular momentum
        for k in range(n_centers):
            for direction in range(3):
                if angmom_tuple[k][direction] > 0:
                    # create lowered angular momentum for center k
                    lowered_k = list(angmom_tuple[k])
                    lowered_k[direction] -= 1
                    lowered_k = tuple(lowered_k)

                    lowered_tuple = list(angmom_tuple)
                    lowered_tuple[k] = lowered_k
                    lowered_tuple = tuple(lowered_tuple)

                    # first term: (P_i - A_i^(k)) * S(lowered)
                    val = (center_p[direction] - coords[k, direction]) * get_integral(lowered_tuple)

                    # second term: sum over ALL centers
                    for m in range(n_centers):
                        a_m_i = lowered_tuple[m][direction]
                        if a_m_i > 0:
                            double_lowered = list(lowered_tuple)
                            lowered_m = list(double_lowered[m])
                            lowered_m[direction] -= 1
                            double_lowered[m] = tuple(lowered_m)
                            double_lowered = tuple(double_lowered)

                            val += (a_m_i / (2 * gamma)) * get_integral(double_lowered)

                    cache[angmom_tuple] = val
                    return val

        raise RuntimeError("Unexpected state in angular momentum recursion")

    # compute all required integrals
    for idx in np.ndindex(output_shape):
        angmom_tuple = tuple(tuple(angmom_comps_list[k][idx[k]]) for k in range(n_centers))
        integrals[idx] = get_integral(angmom_tuple)

    return integrals


def _compute_n_center_primitive_overlap(coords, exponents, angmom_comps_list, tol_screen=1e-8):
    r"""Compute N-center overlap integral for primitive Gaussians.

    Parameters
    ----------
    coords : np.ndarray(N, 3)
        Coordinates of the N Gaussian centers.
    exponents : np.ndarray(N,)
        Exponents of the N primitive Gaussians.
    angmom_comps_list : list of np.ndarray
        List of N arrays of angular momentum components.
    tol_screen : float, optional
        Screening tolerance. Default is 1e-8.

    Returns
    -------
    integrals : np.ndarray
        Array of overlap integrals.

    """
    gamma, center_p, factor_k = _compute_gaussian_product_params(coords, exponents)

    # screen based on K factor
    if factor_k < tol_screen:
        output_shape = tuple(len(comps) for comps in angmom_comps_list)
        return np.zeros(output_shape)

    # compute via recursion
    return _build_angular_momentum_recursion(
        coords, angmom_comps_list, gamma, center_p, factor_k
    )
