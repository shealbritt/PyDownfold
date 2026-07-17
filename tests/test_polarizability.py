from __future__ import annotations

import numpy as np
import pytest

from pydownfold.polarizability import (
    build_polarizability,
    exclude_active_transitions,
    particle_hole_pairs,
    polarization_weights,
    transition_energies,
)


def test_particle_hole_pairs_cartesian_product_order():
    occupied = [0, 1]
    virtuals = [3, 4]

    transitions = particle_hole_pairs(
        occupied,
        virtuals,
    )

    assert transitions == (
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 4),
    )


def test_particle_hole_pairs_returns_empty_for_empty_occupied_space():
    transitions = particle_hole_pairs(
        occupied=[],
        virtuals=[2, 3],
    )

    assert transitions == ()


def test_particle_hole_pairs_returns_empty_for_empty_virtual_space():
    transitions = particle_hole_pairs(
        occupied=[0, 1],
        virtuals=[],
    )

    assert transitions == ()


def test_particle_hole_pairs_converts_indices_to_int():
    transitions = particle_hole_pairs(
        occupied=np.asarray([0, 1], dtype=np.int64),
        virtuals=np.asarray([3, 4], dtype=np.int64),
    )

    assert transitions == (
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 4),
    )

    assert all(
        isinstance(index, int)
        for transition in transitions
        for index in transition
    )


def test_exclude_active_transitions_removes_only_active_active_pairs():
    transitions = (
        (0, 3),
        (1, 3),
        (1, 4),
        (2, 5),
    )
    active_orbitals = [1, 3, 4]

    residual = exclude_active_transitions(
        transitions,
        active_orbitals,
    )

    assert residual == (
        (0, 3),
        (2, 5),
    )


def test_exclude_active_transitions_preserves_order():
    transitions = (
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )

    residual = exclude_active_transitions(
        transitions,
        active_orbitals=[1, 5, 3],
    )

    assert residual == (
        (0, 4),
        (2, 6),
        (3, 7),
    )


def test_exclude_active_transitions_with_empty_active_space():
    transitions = (
        (0, 3),
        (1, 4),
    )

    residual = exclude_active_transitions(
        transitions,
        active_orbitals=[],
    )

    assert residual == transitions


def test_exclude_active_transitions_with_empty_transition_list():
    residual = exclude_active_transitions(
        transitions=[],
        active_orbitals=[0, 1],
    )

    assert residual == ()


def test_transition_energies_returns_virtual_minus_occupied():
    mo_energies = np.asarray(
        [-1.2, -0.7, -0.1, 0.4, 0.9],
        dtype=float,
    )
    transitions = (
        (0, 3),
        (1, 4),
        (2, 3),
    )

    actual = transition_energies(
        transitions,
        mo_energies,
    )

    expected = np.asarray(
        [
            0.4 - (-1.2),
            0.9 - (-0.7),
            0.4 - (-0.1),
        ]
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
    )


def test_transition_energies_preserves_transition_order():
    mo_energies = np.asarray(
        [-1.0, -0.5, 0.2, 0.8],
        dtype=float,
    )

    forward = transition_energies(
        [(0, 2), (1, 3)],
        mo_energies,
    )

    reversed_result = transition_energies(
        [(1, 3), (0, 2)],
        mo_energies,
    )

    np.testing.assert_allclose(
        forward,
        reversed_result[::-1],
        atol=0.0,
        rtol=0.0,
    )


def test_transition_energies_empty_transition_list():
    actual = transition_energies(
        transitions=[],
        mo_energies=np.asarray([-1.0, 0.5]),
    )

    assert actual.shape == (0,)
    assert actual.dtype == np.float64


def test_transition_energies_rejects_zero_gap():
    mo_energies = np.asarray(
        [-1.0, 0.2, 0.2],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        transition_energies(
            transitions=[(1, 2)],
            mo_energies=mo_energies,
        )


def test_transition_energies_rejects_negative_gap():
    mo_energies = np.asarray(
        [-1.0, 0.5, 0.1],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        transition_energies(
            transitions=[(1, 2)],
            mo_energies=mo_energies,
        )

def test_polarization_weights_matches_definition():
    gaps = np.asarray(
        [0.5, 1.0, 1.5],
        dtype=float,
    )
    omega = 0.2
    eta = 1e-4

    actual = polarization_weights(
        gaps,
        omega=omega,
        eta=eta,
        closed_shell=False,
    )

    expected = (
        1.0 / (omega - gaps + 1j * eta)
        - 1.0 / (omega + gaps - 1j * eta)
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-14,
        rtol=1e-14,
    )

def test_polarization_weights_closed_shell_static_limit():
    gaps = np.asarray(
        [0.5, 1.0, 2.0],
        dtype=float,
    )

    actual = polarization_weights(
        gaps,
        omega=0.0,
        eta=1e-10,
        closed_shell=True,
    )

    expected = -4.0 / gaps

    np.testing.assert_allclose(
        actual.real,
        expected,
        atol=1e-10,
        rtol=1e-10,
    )

def test_polarization_weights_static_limit():
    gaps = np.asarray(
        [0.5, 1.0, 2.0],
        dtype=float,
    )
    eta = 1e-10

    actual = polarization_weights(
        gaps,
        omega=0.0,
        eta=eta,
        closed_shell=False,
    )

    expected = (
        -2.0 * gaps / (gaps**2 + eta**2)
        - 2.0j * eta / (gaps**2 + eta**2)
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-14,
        rtol=1e-14,
    )
        
def test_polarization_weights_returns_complex_array():
    actual = polarization_weights(
        np.asarray([0.5, 1.0]),
    )

    assert actual.dtype == np.complex128
    assert np.iscomplexobj(actual)


def test_polarization_weights_empty_input():
    actual = polarization_weights(
        np.asarray([], dtype=float),
    )

    assert actual.shape == (0,)
    assert actual.dtype == np.complex128


def test_polarization_weights_rejects_nonvector_input():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        polarization_weights(
            np.asarray([[0.5, 1.0]]),
        )


def test_polarization_weights_rejects_nonpositive_eta():
    gaps = np.asarray([0.5, 1.0])

    with pytest.raises(
        ValueError,
        match="eta must be positive",
    ):
        polarization_weights(
            gaps,
            eta=0.0,
        )

    with pytest.raises(
        ValueError,
        match="eta must be positive",
    ):
        polarization_weights(
            gaps,
            eta=-1e-5,
        )


def test_build_polarizability_shape():
    product_overlaps = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [0.5, 0.2, 0.1],
        ]
    )
    weights = np.asarray(
        [1.0, 2.0, 3.0],
        dtype=np.complex128,
    )

    polarizability = build_polarizability(
        product_overlaps,
        weights,
    )

    assert polarizability.shape == (2, 2)
    assert polarizability.dtype == np.complex128


def test_build_polarizability_matches_explicit_sum():
    product_overlaps = np.asarray(
        [
            [1.0, 2.0, -1.0],
            [0.5, -0.3, 0.8],
            [2.0, 0.4, 0.1],
        ],
        dtype=float,
    )
    weights = np.asarray(
        [
            1.0 + 0.2j,
            -0.5 + 0.1j,
            0.3 - 0.4j,
        ],
        dtype=np.complex128,
    )

    actual = build_polarizability(
        product_overlaps,
        weights,
    )

    expected = np.zeros(
        (product_overlaps.shape[0], product_overlaps.shape[0]),
        dtype=np.complex128,
    )

    for transition in range(product_overlaps.shape[1]):
        vector = product_overlaps[:, transition]

        expected += (
            weights[transition]
            * np.outer(vector, vector)
        )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-14,
        rtol=1e-14,
    )


def test_build_polarizability_matches_matrix_product():
    product_overlaps = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    weights = np.asarray(
        [0.5 - 0.1j, -2.0 + 0.3j],
        dtype=np.complex128,
    )

    actual = build_polarizability(
        product_overlaps,
        weights,
    )

    expected = (
        product_overlaps
        * weights[None, :]
    ) @ product_overlaps.T

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-14,
        rtol=1e-14,
    )


def test_build_polarizability_is_complex_symmetric():
    product_overlaps = np.asarray(
        [
            [1.0, 0.2, 0.3],
            [0.4, 1.5, 0.1],
            [0.7, 0.8, 2.0],
        ]
    )
    weights = np.asarray(
        [
            1.0 + 0.2j,
            -0.5 + 0.1j,
            0.3 - 0.4j,
        ]
    )

    polarizability = build_polarizability(
        product_overlaps,
        weights,
    )

    np.testing.assert_allclose(
        polarizability,
        polarizability.T,
        atol=1e-14,
        rtol=1e-14,
    )


def test_build_polarizability_static_response_is_negative_semidefinite():
    product_overlaps = np.asarray(
        [
            [1.0, 0.2, -0.4],
            [0.3, 1.5, 0.1],
            [0.7, -0.8, 2.0],
        ]
    )
    gaps = np.asarray(
        [0.5, 1.0, 2.0],
    )

    weights = polarization_weights(
        gaps,
        omega=0.0,
        eta=1e-12,
    )

    polarizability = build_polarizability(
        product_overlaps,
        weights,
    )

    eigenvalues = np.linalg.eigvalsh(
        polarizability.real,
    )

    assert np.all(eigenvalues <= 1e-10)


def test_build_polarizability_zero_transitions():
    product_overlaps = np.empty(
        (4, 0),
        dtype=float,
    )
    weights = np.empty(
        (0,),
        dtype=np.complex128,
    )

    actual = build_polarizability(
        product_overlaps,
        weights,
    )

    expected = np.zeros(
        (4, 4),
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
    )


def test_build_polarizability_rejects_nonmatrix_overlaps():
    with pytest.raises(
        ValueError,
        match="product_overlaps must have shape",
    ):
        build_polarizability(
            product_overlaps=np.asarray([1.0, 2.0]),
            weights=np.asarray([1.0, 2.0]),
        )


def test_build_polarizability_rejects_nonvector_weights():
    with pytest.raises(
        ValueError,
        match="weights must be one-dimensional",
    ):
        build_polarizability(
            product_overlaps=np.ones((2, 2)),
            weights=np.ones((2, 1)),
        )


def test_build_polarizability_rejects_transition_count_mismatch():
    with pytest.raises(
        ValueError,
        match="one value per transition",
    ):
        build_polarizability(
            product_overlaps=np.ones((3, 2)),
            weights=np.ones(3),
        )