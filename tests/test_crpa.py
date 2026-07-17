from __future__ import annotations

import numpy as np
import pytest
from pyscf import gto, scf

import pydownfold.crpa as crpa
from pydownfold.integrals import bare_active_eri


@pytest.fixture(scope="module")
def water_rhf():
    """Return a small converged closed-shell RHF calculation."""
    mol = gto.M(
        atom="""
        O  0.000000  0.000000  0.000000
        H  0.000000 -0.757000  0.587000
        H  0.000000  0.757000  0.587000
        """,
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )

    mf = scf.RHF(mol).run()

    assert mf.converged
    return mf


def test_kernel_returns_active_space_four_index_tensor(water_rhf):
    active_orbitals = [4, 5]

    screened = crpa.kernel(
        water_rhf,
        active_orbitals,
    )

    assert screened.shape == (2, 2, 2, 2)
    assert np.iscomplexobj(screened)


def test_kernel_returns_finite_values(water_rhf):
    screened = crpa.kernel(
        water_rhf,
        active_orbitals=[4, 5],
    )

    assert np.all(np.isfinite(screened.real))
    assert np.all(np.isfinite(screened.imag))


def test_kernel_preserves_pair_exchange_symmetry_at_zero_frequency(
    water_rhf,
):
    screened = crpa.kernel(
        water_rhf,
        active_orbitals=[4, 5],
        omega=0.0,
        eta=1e-5,
    )

    np.testing.assert_allclose(
        screened,
        screened.transpose(2, 3, 0, 1),
        atol=1e-10,
        rtol=1e-10,
    )


def test_kernel_is_nearly_real_at_zero_frequency(water_rhf):
    screened = crpa.kernel(
        water_rhf,
        active_orbitals=[4, 5],
        omega=0.0,
        eta=1e-8,
    )

    np.testing.assert_allclose(
        screened.imag,
        0.0,
        atol=1e-7,
        rtol=0.0,
    )


def test_kernel_screening_changes_bare_interaction(water_rhf):
    active_orbitals = [4, 5]

    bare = bare_active_eri(
        water_rhf,
        active_orbitals,
    )

    screened = crpa.kernel(
        water_rhf,
        active_orbitals,
        omega=0.0,
        eta=1e-5,
    )

    assert not np.allclose(
        screened.real,
        bare,
        atol=1e-12,
        rtol=1e-12,
    )


def test_kernel_matches_explicit_dyson_solution(
    water_rhf,
):
    active_orbitals = [4, 5]
    omega = 0.0
    eta = 1e-5

    mo_occ = np.asarray(water_rhf.mo_occ)
    mo_energy = np.asarray(water_rhf.mo_energy)

    occupied = tuple(
        int(index)
        for index in np.flatnonzero(mo_occ == 2.0)
    )
    virtual = tuple(
        int(index)
        for index in np.flatnonzero(mo_occ == 0.0)
    )

    transitions = crpa.particle_hole_pairs(
        occupied,
        virtual,
    )
    residual_transitions = crpa.exclude_active_transitions(
        transitions,
        active_orbitals,
    )

    gaps = crpa.transition_energies(
        residual_transitions,
        mo_energy,
    )

    weights = crpa.polarization_weights(
        gaps,
        omega=omega,
        eta=eta,
    )

    product_overlaps = crpa.product_overlap_matrix(
        water_rhf,
        pair_orbitals=active_orbitals,
        transitions=residual_transitions,
    )

    polarizability = crpa.build_polarizability(
        product_overlaps,
        weights,
    )

    bare = bare_active_eri(
        water_rhf,
        active_orbitals,
    )

    nact = len(active_orbitals)
    npairs = nact * nact

    v = np.asarray(
        bare,
        dtype=np.complex128,
    ).reshape(npairs, npairs)

    expected = np.linalg.solve(
        np.eye(npairs, dtype=np.complex128)
        - v @ polarizability,
        v,
    ).reshape(
        nact,
        nact,
        nact,
        nact,
    )

    actual = crpa.kernel(
        water_rhf,
        active_orbitals,
        omega=omega,
        eta=eta,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_kernel_excludes_active_active_transitions(
    water_rhf,
    monkeypatch,
):
    active_orbitals = [4, 5]
    captured_transitions = None

    original = crpa.product_overlap_matrix

    def wrapped_product_overlap_matrix(
        mf,
        pair_orbitals,
        transitions,
    ):
        nonlocal captured_transitions
        captured_transitions = tuple(transitions)

        return original(
            mf,
            pair_orbitals,
            transitions,
        )

    monkeypatch.setattr(
        crpa,
        "product_overlap_matrix",
        wrapped_product_overlap_matrix,
    )

    crpa.kernel(
        water_rhf,
        active_orbitals,
    )

    assert captured_transitions is not None

    active = set(active_orbitals)

    assert all(
        not (i in active and a in active)
        for i, a in captured_transitions
    )


def test_kernel_applies_closed_shell_spin_factor(
    water_rhf,
    monkeypatch,
):
    captured_weights = None

    original = crpa.build_polarizability

    def wrapped_build_polarizability(
        product_overlaps,
        weights,
    ):
        nonlocal captured_weights
        captured_weights = np.asarray(weights).copy()

        return original(
            product_overlaps,
            weights,
        )

    monkeypatch.setattr(
        crpa,
        "build_polarizability",
        wrapped_build_polarizability,
    )

    active_orbitals = [4, 5]
    omega = 0.0
    eta = 1e-5

    crpa.kernel(
        water_rhf,
        active_orbitals,
        omega=omega,
        eta=eta,
    )

    mo_occ = np.asarray(water_rhf.mo_occ)
    mo_energy = np.asarray(water_rhf.mo_energy)

    occupied = tuple(
        int(index)
        for index in np.flatnonzero(mo_occ == 2.0)
    )
    virtual = tuple(
        int(index)
        for index in np.flatnonzero(mo_occ == 0.0)
    )

    transitions = crpa.particle_hole_pairs(
        occupied,
        virtual,
    )
    transitions = crpa.exclude_active_transitions(
        transitions,
        active_orbitals,
    )

    gaps = crpa.transition_energies(
        transitions,
        mo_energy,
    )

    expected = crpa.polarization_weights(
        gaps,
        omega=omega,
        eta=eta,
    )

    assert captured_weights is not None

    np.testing.assert_allclose(
        captured_weights,
        expected,
        atol=1e-14,
        rtol=1e-14,
    )


def test_kernel_returns_bare_interaction_when_no_residual_transitions(
    water_rhf,
):
    nmo = np.asarray(water_rhf.mo_coeff).shape[1]
    active_orbitals = list(range(nmo))

    expected = bare_active_eri(
        water_rhf,
        active_orbitals,
    ).astype(np.complex128)

    actual = crpa.kernel(
        water_rhf,
        active_orbitals,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_kernel_rejects_unconverged_mean_field(water_rhf):
    original = water_rhf.converged
    water_rhf.converged = False

    try:
        with pytest.raises(
            ValueError,
            match="must be converged",
        ):
            crpa.kernel(
                water_rhf,
                active_orbitals=[4, 5],
            )
    finally:
        water_rhf.converged = original


def test_kernel_rejects_empty_active_space(water_rhf):
    with pytest.raises(
        ValueError,
        match="at least one orbital",
    ):
        crpa.kernel(
            water_rhf,
            active_orbitals=[],
        )


def test_kernel_rejects_duplicate_active_orbitals(water_rhf):
    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        crpa.kernel(
            water_rhf,
            active_orbitals=[4, 4],
        )


@pytest.mark.parametrize(
    "active_orbitals",
    [
        [-1, 4],
        [4, 7],
    ],
)
def test_kernel_rejects_out_of_range_active_orbitals(
    water_rhf,
    active_orbitals,
):
    with pytest.raises(
        IndexError,
        match="outside the MO range",
    ):
        crpa.kernel(
            water_rhf,
            active_orbitals,
        )


def test_kernel_rejects_non_closed_shell_occupations(
    water_rhf,
):
    original = np.asarray(water_rhf.mo_occ).copy()

    modified = original.copy()
    modified[4] = 1.0
    water_rhf.mo_occ = modified

    try:
        with pytest.raises(
            ValueError,
            match="occupations of 0 or 2",
        ):
            crpa.kernel(
                water_rhf,
                active_orbitals=[4, 5],
            )
    finally:
        water_rhf.mo_occ = original