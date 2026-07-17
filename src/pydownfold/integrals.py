from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pyscf import ao2mo


FloatArray = NDArray[np.float64]

def bare_active_eri(
    mf,
    active_orbitals: Sequence[int],
) -> FloatArray:
    """Return full four-index bare ERIs in the active MO basis."""
    if isinstance(mf.mo_coeff, tuple):
        raise NotImplementedError(
            "bare_active_eri currently supports restricted references only."
        )

    active = tuple(int(index) for index in active_orbitals)
    coeff = np.asarray(mf.mo_coeff)[:, active]

    return mo_eri_block(
        mf,
        (coeff, coeff, coeff, coeff),
    )


def mo_eri_block(
    mf,
    coeffs: Sequence[FloatArray],
) -> FloatArray:
    """Return a four-index MO ERI block in chemist's notation.

    Parameters
    ----------
    mf
        Converged restricted PySCF mean-field object.
    coeffs
        Four MO-coefficient matrices ``(C_p, C_q, C_r, C_s)``.
        Each matrix must have shape ``(nao, n_orbitals)``.

    Returns
    -------
    numpy.ndarray
        Tensor with shape ``(n_p, n_q, n_r, n_s)`` containing
        ``(pq|rs)`` integrals in chemist's notation.
    """
    if len(coeffs) != 4:
        raise ValueError("coeffs must contain exactly four coefficient matrices.")

    arrays = tuple(np.asarray(coeff) for coeff in coeffs)

    nao = mf.mol.nao_nr()
    for index, coeff in enumerate(arrays):
        if coeff.ndim != 2:
            raise ValueError(f"Coefficient matrix {index} must be two-dimensional.")
        if coeff.shape[0] != nao:
            raise ValueError(
                f"Coefficient matrix {index} has {coeff.shape[0]} AO rows; "
                f"expected {nao}."
            )

    dimensions = tuple(coeff.shape[1] for coeff in arrays)

    eri = ao2mo.general(
        mf.mol,
        arrays,
        compact=False,
    )

    return np.asarray(eri).reshape(dimensions)


def particle_hole_eri(
    mf,
    occupied: Sequence[int],
    virtual: Sequence[int],
) -> FloatArray:
    """Return the particle-hole Coulomb block ``(ia|jb)``.

    Parameters
    ----------
    mf
        Converged restricted PySCF mean-field object.
    occupied
        Indices of occupied molecular orbitals.
    virtual
        Indices of virtual molecular orbitals.

    Returns
    -------
    numpy.ndarray
        Tensor with shape ``(nocc, nvir, nocc, nvir)`` ordered as
        ``eri[i, a, j, b] = (ia|jb)``.
    """
    if isinstance(mf.mo_coeff, tuple):
        raise NotImplementedError(
            "particle_hole_eri currently supports restricted references only."
        )

    mo_coeff = np.asarray(mf.mo_coeff)

    if mo_coeff.ndim != 2:
        raise ValueError("mf.mo_coeff must be a two-dimensional array.")

    occupied = tuple(int(index) for index in occupied)
    virtual = tuple(int(index) for index in virtual)

    nmo = mo_coeff.shape[1]

    if len(set(occupied)) != len(occupied):
        raise ValueError("Occupied orbital indices must be unique.")

    if len(set(virtual)) != len(virtual):
        raise ValueError("Virtual orbital indices must be unique.")

    all_indices = occupied + virtual
    if any(index < 0 or index >= nmo for index in all_indices):
        raise IndexError("An orbital index is outside the MO range.")

    if set(occupied) & set(virtual):
        raise ValueError("Occupied and virtual orbital sets must not overlap.")

    coeff_occ = mo_coeff[:, occupied]
    coeff_vir = mo_coeff[:, virtual]

    return mo_eri_block(
        mf,
        (
            coeff_occ,
            coeff_vir,
            coeff_occ,
            coeff_vir,
        ),
    )

def product_overlap_matrix(
    mf,
    pair_orbitals,
    transitions,
):
    """
    Compute

        S[p, q, t]
        = ∫ φ_p(r) φ_q(r) φ_i_t(r) φ_a_t(r) dr

    where t labels the transition (i_t, a_t).
    """
    import numpy as np

    C = np.asarray(mf.mo_coeff)

    pair_orbitals = np.asarray(pair_orbitals, dtype=int)
    transitions = tuple(transitions)

    if C.ndim != 2:
        raise NotImplementedError(
            "Only restricted real-valued orbitals are supported."
        )

    nact = len(pair_orbitals)
    ntrans = len(transitions)

    if ntrans == 0:
        return np.empty((nact * nact, 0), dtype=C.dtype)

    occ = np.asarray([i for i, _ in transitions], dtype=int)
    vir = np.asarray([a for _, a in transitions], dtype=int)

    C_act = C[:, pair_orbitals]
    C_occ = C[:, occ]
    C_vir = C[:, vir]

    # G[μ,ν,κ,λ] = ∫ χ_μ χ_ν χ_κ χ_λ dr
    G_ao = mf.mol.intor("int4c1e", comp=1, aosym="s1")

    # S[p,q,t]
    S = np.einsum(
        "mnkl,mp,nq,kt,lt->pqt",
        G_ao,
        C_act,
        C_act,
        C_occ,
        C_vir,
        optimize=True,
    )

    return S.reshape(nact * nact, ntrans)