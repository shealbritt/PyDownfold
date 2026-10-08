import numpy as np

from pyscf import gto, mcscf, scf

from pydownfold.crpa_screened_casci import run_screened_casci
from pydownfold.crpa_screening import (
    build_screened_hamiltonian,
)


def build_h2(basis):
    mol = gto.M(
        atom="""
        H 0.0 0.0 0.0
        H 0.0 0.0 0.74
        """,
        basis=basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=0,
    )

    mf = scf.RHF(mol).run()

    assert mf.converged

    return mf


def test_screened_hamiltonian_shapes():
    mf = build_h2("cc-pvdz")

    hamiltonian = build_screened_hamiltonian(
        mf,
        active_orbitals=[0, 1],
        nelecas=2,
        omega=0.0,
    )

    assert hamiltonian.h1.shape == (2, 2)
    assert hamiltonian.h2.shape == (2, 2, 2, 2)
    assert hamiltonian.t_dc.shape == (2, 2)

    assert np.all(np.isfinite(hamiltonian.h1))
    assert np.all(np.isfinite(hamiltonian.h2))
    assert np.isfinite(hamiltonian.ecore)


def test_no_double_counting_leaves_h1_unchanged():
    mf = build_h2("cc-pvdz")

    hamiltonian = build_screened_hamiltonian(
        mf,
        active_orbitals=[0, 1],
        nelecas=2,
        omega=0.0,
        apply_double_counting=False,
    )

    assert np.allclose(
        hamiltonian.h1,
        hamiltonian.h1_bare,
    )

    assert np.allclose(
        hamiltonian.t_dc,
        0.0,
    )


def test_all_orbitals_active_recovers_bare_casci():
    """With no external transitions, cRPA screening must vanish."""

    mf = build_h2("sto-3g")

    # STO-3G H2 has exactly two spatial MOs, and both are active.
    hamiltonian = build_screened_hamiltonian(
        mf,
        active_orbitals=[0, 1],
        nelecas=2,
        omega=0.0,
    )

    bare = mcscf.CASCI(
        mf,
        2,
        2,
    )

    e_bare, _ = bare.kernel()

    energies, _ = run_screened_casci(
        mf,
        nelecas=2,
        hamiltonian=hamiltonian,
    )

    assert np.allclose(
        hamiltonian.h2,
        hamiltonian.h2_bare,
        atol=1e-9,
    )

    assert np.allclose(
        energies[0],
        e_bare,
        atol=1e-9,
    )


def test_screened_casci_returns_finite_energy():
    mf = build_h2("cc-pvdz")

    hamiltonian = build_screened_hamiltonian(
        mf,
        active_orbitals=[0, 1],
        nelecas=2,
        omega=0.0,
    )

    energies, cas = run_screened_casci(
        mf,
        nelecas=2,
        hamiltonian=hamiltonian,
    )

    assert energies.shape == (1,)
    assert np.isfinite(energies[0])
    assert cas.converged