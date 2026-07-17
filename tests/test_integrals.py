from __future__ import annotations

import numpy as np
import pytest
from pyscf import ao2mo, gto, scf

from pydownfold.integrals import (
    bare_active_eri,
    mo_eri_block,
    particle_hole_eri,
    product_overlap_matrix,
)


@pytest.fixture(scope="module")
def water_rhf():
    """Return a small converged restricted Hartree–Fock calculation."""
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


def test_mo_eri_block_shape(water_rhf):
    coeff = np.asarray(water_rhf.mo_coeff)

    coeff_p = coeff[:, [0, 1]]
    coeff_q = coeff[:, [2, 3, 4]]
    coeff_r = coeff[:, [1]]
    coeff_s = coeff[:, [5, 6]]

    eri = mo_eri_block(
        water_rhf,
        (
            coeff_p,
            coeff_q,
            coeff_r,
            coeff_s,
        ),
    )

    assert eri.shape == (2, 3, 1, 2)


def test_mo_eri_block_matches_ao2mo_general(water_rhf):
    coeff = np.asarray(water_rhf.mo_coeff)

    coeff_p = coeff[:, [0, 1]]
    coeff_q = coeff[:, [2, 3]]
    coeff_r = coeff[:, [1, 4]]
    coeff_s = coeff[:, [5, 6]]

    actual = mo_eri_block(
        water_rhf,
        (
            coeff_p,
            coeff_q,
            coeff_r,
            coeff_s,
        ),
    )

    expected = ao2mo.general(
        water_rhf.mol,
        (
            coeff_p,
            coeff_q,
            coeff_r,
            coeff_s,
        ),
        compact=False,
    ).reshape(2, 2, 2, 2)

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_mo_eri_block_requires_four_coefficient_matrices(water_rhf):
    coeff = np.asarray(water_rhf.mo_coeff)

    with pytest.raises(
        ValueError,
        match="exactly four coefficient matrices",
    ):
        mo_eri_block(
            water_rhf,
            (
                coeff,
                coeff,
                coeff,
            ),
        )


def test_mo_eri_block_rejects_nonmatrix_coefficients(water_rhf):
    coeff = np.asarray(water_rhf.mo_coeff)

    with pytest.raises(
        ValueError,
        match="must be two-dimensional",
    ):
        mo_eri_block(
            water_rhf,
            (
                coeff[:, 0],
                coeff,
                coeff,
                coeff,
            ),
        )


def test_mo_eri_block_rejects_wrong_number_of_ao_rows(water_rhf):
    coeff = np.asarray(water_rhf.mo_coeff)
    invalid = coeff[:-1]

    with pytest.raises(
        ValueError,
        match="AO rows",
    ):
        mo_eri_block(
            water_rhf,
            (
                invalid,
                coeff,
                coeff,
                coeff,
            ),
        )


def test_bare_active_eri_shape(water_rhf):
    active_orbitals = [2, 3, 4]

    eri = bare_active_eri(
        water_rhf,
        active_orbitals,
    )

    assert eri.shape == (3, 3, 3, 3)


def test_bare_active_eri_matches_direct_transformation(water_rhf):
    active_orbitals = [2, 3, 4]
    coeff = np.asarray(water_rhf.mo_coeff)[:, active_orbitals]

    actual = bare_active_eri(
        water_rhf,
        active_orbitals,
    )

    expected = ao2mo.general(
        water_rhf.mol,
        (
            coeff,
            coeff,
            coeff,
            coeff,
        ),
        compact=False,
    ).reshape(3, 3, 3, 3)

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_bare_active_eri_permutational_symmetry(water_rhf):
    active_orbitals = [2, 3, 4]

    eri = bare_active_eri(
        water_rhf,
        active_orbitals,
    )

    np.testing.assert_allclose(
        eri,
        eri.swapaxes(0, 1),
        atol=1e-12,
        rtol=1e-12,
    )

    np.testing.assert_allclose(
        eri,
        eri.swapaxes(2, 3),
        atol=1e-12,
        rtol=1e-12,
    )

    np.testing.assert_allclose(
        eri,
        eri.transpose(2, 3, 0, 1),
        atol=1e-12,
        rtol=1e-12,
    )


def test_particle_hole_eri_shape(water_rhf):
    occupied = [0, 1]
    virtual = [5, 6]

    eri = particle_hole_eri(
        water_rhf,
        occupied,
        virtual,
    )

    assert eri.shape == (2, 2, 2, 2)


def test_particle_hole_eri_matches_direct_transformation(water_rhf):
    occupied = [0, 1]
    virtual = [5, 6]

    coeff = np.asarray(water_rhf.mo_coeff)
    coeff_occ = coeff[:, occupied]
    coeff_vir = coeff[:, virtual]

    actual = particle_hole_eri(
        water_rhf,
        occupied,
        virtual,
    )

    expected = ao2mo.general(
        water_rhf.mol,
        (
            coeff_occ,
            coeff_vir,
            coeff_occ,
            coeff_vir,
        ),
        compact=False,
    ).reshape(
        len(occupied),
        len(virtual),
        len(occupied),
        len(virtual),
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_particle_hole_eri_pair_exchange_symmetry(water_rhf):
    occupied = [0, 1]
    virtual = [5, 6]

    eri = particle_hole_eri(
        water_rhf,
        occupied,
        virtual,
    )

    np.testing.assert_allclose(
        eri,
        eri.transpose(2, 3, 0, 1),
        atol=1e-12,
        rtol=1e-12,
    )


def test_particle_hole_eri_rejects_duplicate_occupied_indices(
    water_rhf,
):
    with pytest.raises(
        ValueError,
        match="Occupied orbital indices must be unique",
    ):
        particle_hole_eri(
            water_rhf,
            occupied=[0, 0],
            virtual=[5, 6],
        )


def test_particle_hole_eri_rejects_duplicate_virtual_indices(
    water_rhf,
):
    with pytest.raises(
        ValueError,
        match="Virtual orbital indices must be unique",
    ):
        particle_hole_eri(
            water_rhf,
            occupied=[0, 1],
            virtual=[5, 5],
        )


def test_particle_hole_eri_rejects_overlapping_spaces(water_rhf):
    with pytest.raises(
        ValueError,
        match="must not overlap",
    ):
        particle_hole_eri(
            water_rhf,
            occupied=[0, 1],
            virtual=[1, 6],
        )


def test_particle_hole_eri_rejects_out_of_range_index(water_rhf):
    nmo = np.asarray(water_rhf.mo_coeff).shape[1]

    with pytest.raises(
        IndexError,
        match="outside the MO range",
    ):
        particle_hole_eri(
            water_rhf,
            occupied=[0, 1],
            virtual=[5, nmo],
        )


def test_product_overlap_matrix_shape(water_rhf):
    pair_orbitals = [2, 3, 4]
    transitions = [
        (0, 5),
        (1, 6),
        (4, 6),
    ]

    overlap = product_overlap_matrix(
        water_rhf,
        pair_orbitals,
        transitions,
    )

    assert overlap.shape == (
        len(pair_orbitals) ** 2,
        len(transitions),
    )


def test_product_overlap_matrix_empty_transitions(water_rhf):
    pair_orbitals = [2, 3, 4]

    overlap = product_overlap_matrix(
        water_rhf,
        pair_orbitals,
        transitions=[],
    )

    assert overlap.shape == (
        len(pair_orbitals) ** 2,
        0,
    )


def test_product_overlap_matrix_pair_symmetry(water_rhf):
    pair_orbitals = [2, 3, 4]
    transitions = [
        (0, 5),
        (1, 6),
        (4, 6),
    ]

    overlap = product_overlap_matrix(
        water_rhf,
        pair_orbitals,
        transitions,
    )

    overlap_tensor = overlap.reshape(
        len(pair_orbitals),
        len(pair_orbitals),
        len(transitions),
    )

    np.testing.assert_allclose(
        overlap_tensor,
        overlap_tensor.swapaxes(0, 1),
        atol=1e-12,
        rtol=1e-12,
    )


def test_product_overlap_matrix_matches_explicit_ao_transformation(
    water_rhf,
):
    pair_orbitals = [2, 3, 4]
    transitions = [
        (0, 5),
        (1, 6),
        (4, 6),
    ]

    actual = product_overlap_matrix(
        water_rhf,
        pair_orbitals,
        transitions,
    ).reshape(
        len(pair_orbitals),
        len(pair_orbitals),
        len(transitions),
    )

    coeff = np.asarray(water_rhf.mo_coeff)

    coeff_pair = coeff[:, pair_orbitals]
    occupied = np.asarray(
        [i for i, _ in transitions],
        dtype=int,
    )
    virtual = np.asarray(
        [a for _, a in transitions],
        dtype=int,
    )

    coeff_occ = coeff[:, occupied]
    coeff_vir = coeff[:, virtual]

    ao_product_overlap = water_rhf.mol.intor(
        "int4c1e",
        comp=1,
        aosym="s1",
    )

    expected = np.einsum(
        "mnkl,mp,nq,kt,lt->pqt",
        ao_product_overlap,
        coeff_pair,
        coeff_pair,
        coeff_occ,
        coeff_vir,
        optimize=True,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_product_overlap_matrix_single_element(
    water_rhf,
):
    pair_orbitals = [2, 3, 4]
    transitions = [
        (0, 5),
        (1, 6),
        (4, 6),
    ]

    overlap = product_overlap_matrix(
        water_rhf,
        pair_orbitals,
        transitions,
    ).reshape(
        len(pair_orbitals),
        len(pair_orbitals),
        len(transitions),
    )

    p_position = 1
    q_position = 2
    transition_position = 1

    p = pair_orbitals[p_position]
    q = pair_orbitals[q_position]
    i, a = transitions[transition_position]

    coeff = np.asarray(water_rhf.mo_coeff)
    ao_product_overlap = water_rhf.mol.intor(
        "int4c1e",
        comp=1,
        aosym="s1",
    )

    expected = np.einsum(
        "mnkl,m,n,k,l->",
        ao_product_overlap,
        coeff[:, p],
        coeff[:, q],
        coeff[:, i],
        coeff[:, a],
        optimize=True,
    )

    np.testing.assert_allclose(
        overlap[p_position, q_position, transition_position],
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_product_overlap_transition_order_is_preserved(
    water_rhf,
):
    pair_orbitals = [2, 3]

    transitions = [
        (0, 5),
        (1, 6),
    ]

    reversed_transitions = list(reversed(transitions))

    forward = product_overlap_matrix(
        water_rhf,
        pair_orbitals,
        transitions,
    )

    reversed_result = product_overlap_matrix(
        water_rhf,
        pair_orbitals,
        reversed_transitions,
    )

    np.testing.assert_allclose(
        forward[:, 0],
        reversed_result[:, 1],
        atol=1e-12,
        rtol=1e-12,
    )

    np.testing.assert_allclose(
        forward[:, 1],
        reversed_result[:, 0],
        atol=1e-12,
        rtol=1e-12,
    )