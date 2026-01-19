"""Test gbasis.integrals.overlap_n_center."""

import numpy as np
import pytest
from gbasis.integrals._overlap_n_center import (
    _compute_gaussian_product_params,
    _compute_primitive_s_overlap,
)
from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.overlap_n_center import n_center_overlap_integral
from gbasis.parsers import make_contractions, parse_nwchem
from gbasis.screening import is_n_center_overlap_screened
from utils import find_datafile


def test_gaussian_product_params_two_centers():
    """Test _compute_gaussian_product_params for two centers."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    exponents = np.array([1.0, 2.0])

    gamma, center_p, factor_k = _compute_gaussian_product_params(coords, exponents)

    assert np.isclose(gamma, 3.0)
    assert np.allclose(center_p, [2 / 3, 0.0, 0.0])
    assert np.isclose(factor_k, np.exp(-2 / 3))


def test_gaussian_product_params_three_centers():
    """Test _compute_gaussian_product_params for three centers."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    exponents = np.array([1.0, 1.0, 1.0])

    gamma, center_p, factor_k = _compute_gaussian_product_params(coords, exponents)

    assert np.isclose(gamma, 3.0)
    assert np.allclose(center_p, [1 / 3, 1 / 3, 0.0])
    expected_k = np.exp(-(1 + 1 + 2) / 3)
    assert np.isclose(factor_k, expected_k)


def test_primitive_s_overlap_same_center():
    """Test _compute_primitive_s_overlap when centers coincide."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    exponents = np.array([1.0, 1.0])

    overlap_val = _compute_primitive_s_overlap(coords, exponents)

    expected = (np.pi / 2) ** 1.5
    assert np.isclose(overlap_val, expected)


def test_primitive_s_overlap_decay():
    """Test _compute_primitive_s_overlap decay with separation."""
    coords_close = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    coords_far = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    exponents = np.array([1.0, 1.0])

    overlap_close = _compute_primitive_s_overlap(coords_close, exponents)
    overlap_far = _compute_primitive_s_overlap(coords_far, exponents)

    assert overlap_close > overlap_far
    assert overlap_far < 1e-20


def test_screening_close_centers():
    """Test that close centers are not screened."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    basis = make_contractions(basis_dict, ["H", "H"], coords, "cartesian")

    assert not is_n_center_overlap_screened([basis[0], basis[1]], 1e-8)


def test_screening_far_centers():
    """Test that far centers are screened."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    basis = make_contractions(basis_dict, ["H", "H"], coords, "cartesian")

    assert is_n_center_overlap_screened([basis[0], basis[1]], 1e-8)


def test_screening_three_centers():
    """Test screening for three centers."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [100.0, 0.0, 0.0]])
    basis = make_contractions(basis_dict, ["H", "H", "H"], coords, "cartesian")

    assert is_n_center_overlap_screened([basis[0], basis[1], basis[2]], 1e-8)


def test_n2_matches_existing_cartesian():
    """Test that n_center_overlap_integral(n=2) matches overlap_integral for Cartesian."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    basis = make_contractions(basis_dict, ["H", "H"], coords, "cartesian")

    existing_overlap = overlap_integral(basis, screen_basis=False)
    new_overlap = n_center_overlap_integral(basis, n=2, screen_basis=False)

    assert np.allclose(existing_overlap, new_overlap), (
        f"N=2 overlap does not match existing!\n"
        f"Max difference: {np.max(np.abs(existing_overlap - new_overlap))}"
    )


def test_n_center_overlap_type_error_n():
    """Test that TypeError is raised for invalid n type."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["H"], np.array([[0, 0, 0]]), "cartesian")

    with pytest.raises(TypeError):
        n_center_overlap_integral(basis, n=2.5)

    with pytest.raises(TypeError):
        n_center_overlap_integral(basis, n="3")


def test_n_center_overlap_value_error_n():
    """Test that ValueError is raised for invalid n value."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    basis = make_contractions(basis_dict, ["H"], np.array([[0, 0, 0]]), "cartesian")

    with pytest.raises(ValueError):
        n_center_overlap_integral(basis, n=0)

    with pytest.raises(ValueError):
        n_center_overlap_integral(basis, n=-1)


def test_n_center_overlap_type_error_basis():
    """Test that TypeError is raised for invalid basis type."""
    with pytest.raises(TypeError):
        n_center_overlap_integral("not a list", n=2)

    with pytest.raises(TypeError):
        n_center_overlap_integral(["not a shell"], n=2)
