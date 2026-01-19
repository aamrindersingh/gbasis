"""Functions for computing N-center overlap integrals of a basis set."""

from gbasis.contractions import GeneralizedContractionShell
from gbasis.integrals._overlap_n_center import (
    _compute_gaussian_product_params,
    _compute_n_center_primitive_overlap,
)
from gbasis.screening import is_n_center_overlap_screened
from gbasis.spherical import generate_transformation
import numpy as np


def _construct_n_center_overlap_contraction(shells, screen_basis=True, tol_screen=1e-8):
    r"""Compute N-center overlap for a tuple of contraction shells.

    Parameters
    ----------
    shells : tuple of GeneralizedContractionShell
        The N shells to compute the overlap for.
    screen_basis : bool, optional
        Whether to apply screening. Default is True.
    tol_screen : float, optional
        Screening tolerance. Default is 1e-8.

    Returns
    -------
    overlap : np.ndarray
        The overlap integrals with shape (M_0, L_cart_0, M_1, L_cart_1, ...).

    Raises
    ------
    TypeError
        If any shell is not a GeneralizedContractionShell instance.

    """
    for i, shell in enumerate(shells):
        if not isinstance(shell, GeneralizedContractionShell):
            raise TypeError(
                f"Shell at index {i} must be a `GeneralizedContractionShell` instance."
            )

    n_centers = len(shells)

    # determine output shape
    output_shape = []
    for shell in shells:
        output_shape.extend([shell.num_seg_cont, shell.num_cart])
    output_shape = tuple(output_shape)

    # screen if enabled
    if screen_basis and is_n_center_overlap_screened(list(shells), tol_screen):
        return np.zeros(output_shape, dtype=np.float64)

    # get coordinates and angular momentum components
    coords = np.array([shell.coord for shell in shells])
    angmom_comps_list = [shell.angmom_components_cart for shell in shells]

    # initialize output array
    result = np.zeros(output_shape, dtype=np.float64)

    # loop over all primitive combinations
    prim_ranges = tuple(len(shell.exps) for shell in shells)
    for prim_indices in np.ndindex(prim_ranges):
        # get exponents for this primitive combination
        exponents = np.array([shells[k].exps[prim_indices[k]] for k in range(n_centers)])

        # screen at primitive level
        _, _, factor_k = _compute_gaussian_product_params(coords, exponents)
        if factor_k < tol_screen:
            continue

        # compute primitive overlap
        prim_overlap = _compute_n_center_primitive_overlap(
            coords, exponents, angmom_comps_list, tol_screen
        )

        # loop over segmented contractions
        seg_ranges = tuple(shell.num_seg_cont for shell in shells)
        for seg_indices in np.ndindex(seg_ranges):
            # get coefficient product
            coeff_product = 1.0
            for k in range(n_centers):
                coeff_product *= shells[k].coeffs[prim_indices[k], seg_indices[k]]

            # build the index into the result array
            result_index = []
            for k in range(n_centers):
                result_index.extend([seg_indices[k], slice(None)])
            result_index = tuple(result_index)

            # add contribution with normalization
            contrib = coeff_product * prim_overlap
            for k in range(n_centers):
                # norm_prim_cart has shape (num_cart, num_prims)
                norm = shells[k].norm_prim_cart[:, prim_indices[k]]
                shape = [1] * len(prim_overlap.shape)
                shape[k] = len(norm)
                contrib = contrib * norm.reshape(shape)

            result[result_index] += contrib

    return result


def n_center_overlap_integral(basis, n=2, transform=None, screen_basis=True, tol_screen=1e-8):
    r"""Return N-center overlap integral of the given basis set.

    Computes the overlap of N Gaussian basis functions:

    .. math::
        S_{i_1 i_2 ... i_N} = \int \phi_{i_1}(\mathbf{r}) \phi_{i_2}(\mathbf{r})
            \cdots \phi_{i_N}(\mathbf{r}) \, d\mathbf{r}

    Parameters
    ----------
    basis : list/tuple of GeneralizedContractionShell
        Shells of generalized contractions.
    n : int, optional
        Number of centers (N). Must be a positive integer. Default is 2.
    transform : np.ndarray, optional
        Transformation matrix for linear combinations. Default is None.
    screen_basis : bool, optional
        Toggle to enable/disable screening. Default is True.
    tol_screen : float, optional
        Screening tolerance. Default is 1e-8.

    Returns
    -------
    overlap : np.ndarray
        N-center overlap integrals with shape (K, K, ..., K) with N dimensions.

    Raises
    ------
    TypeError
        If `basis` is not a list/tuple of GeneralizedContractionShell.
        If `n` is not an integer.
    ValueError
        If `n` is not a positive integer.

    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError("`n` must be an integer.")
    if n < 1:
        raise ValueError("`n` must be a positive integer.")

    if not isinstance(basis, (list, tuple)):
        raise TypeError("`basis` must be a list or tuple of GeneralizedContractionShell.")
    for shell in basis:
        if not isinstance(shell, GeneralizedContractionShell):
            raise TypeError(
                "All elements of `basis` must be GeneralizedContractionShell instances."
            )

    coord_types = [shell.coord_type for shell in basis]

    # compute basis function counts per shell
    basis_sizes = []
    for shell in basis:
        if shell.coord_type == "spherical":
            size = shell.num_seg_cont * shell.num_sph
        else:
            size = shell.num_seg_cont * shell.num_cart
        basis_sizes.append(size)

    # compute starting indices for each shell
    shell_starts = [0]
    for size in basis_sizes[:-1]:
        shell_starts.append(shell_starts[-1] + size)
    total_basis = sum(basis_sizes)

    # initialize output array
    output_shape = tuple([total_basis] * n)
    result = np.zeros(output_shape, dtype=np.float64)

    # iterate over all shell combinations
    num_shells = len(basis)
    for shell_indices in np.ndindex(tuple([num_shells] * n)):
        shells = tuple(basis[i] for i in shell_indices)

        # compute the block
        block = _construct_n_center_overlap_contraction(
            shells, screen_basis=screen_basis, tol_screen=tol_screen
        )

        # apply contraction normalization
        for k in range(n):
            norm = shells[k].norm_cont
            shape = [1] * (2 * n)
            shape[2 * k] = norm.shape[0]
            shape[2 * k + 1] = norm.shape[1]
            block = block * norm.reshape(shape)

        # apply spherical transformation if needed
        for k in range(n):
            if coord_types[shell_indices[k]] == "spherical":
                transform_sph = generate_transformation(
                    shells[k].angmom,
                    shells[k].angmom_components_cart,
                    shells[k].angmom_components_sph,
                    "left",
                )
                # apply transformation to axis 2*k + 1
                block = np.tensordot(transform_sph, block, (1, 2 * k + 1))
                # move the new axis to the correct position
                block = np.moveaxis(block, 0, 2 * k + 1)

        # flatten the block
        block_shape = []
        for k in range(n):
            if coord_types[shell_indices[k]] == "spherical":
                block_shape.append(shells[k].num_seg_cont * shells[k].num_sph)
            else:
                block_shape.append(shells[k].num_seg_cont * shells[k].num_cart)
        block = block.reshape(block_shape)

        # compute index ranges for placement
        ranges = []
        for k in range(n):
            start = shell_starts[shell_indices[k]]
            end = start + basis_sizes[shell_indices[k]]
            ranges.append(slice(start, end))

        result[tuple(ranges)] = block

    # apply transform if provided
    if transform is not None:
        for k in range(n):
            result = np.tensordot(transform, result, (1, k))
            result = np.moveaxis(result, 0, k)

    return result
