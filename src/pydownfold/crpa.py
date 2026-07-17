from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .integrals import (
    bare_active_eri,
    product_overlap_matrix,
)
from .polarizability import (
    build_polarizability,
    exclude_active_transitions,
    particle_hole_pairs,
    polarization_weights,
    transition_energies,
)


ComplexArray = NDArray[np.complex128]


def kernel(
    mf,
    active_orbitals: Sequence[int],
    omega: float = 0.0,
    eta: float = 1e-5,
) -> ComplexArray:
    """Compute the cRPA-screened active-space interaction.

    The residual polarizability excludes particle-hole transitions
    wholly contained in the active space. For a closed-shell
    restricted reference, each spatial-orbital transition represents
    equal alpha- and beta-spin contributions, so the response weights
    are multiplied by two.

    Parameters
    ----------
    mf
        Converged restricted PySCF mean-field object.
    active_orbitals
        Molecular-orbital indices defining the active space.
    omega
        Screening frequency in Hartree.
    eta
        Positive broadening parameter in Hartree.

    Returns
    -------
    numpy.ndarray
        Screened interaction tensor with shape
        ``(nact, nact, nact, nact)`` in chemist's notation.
    """
    if not getattr(mf, "converged", False):
        raise ValueError(
            "The mean-field calculation must be converged."
        )

    if isinstance(mf.mo_coeff, tuple):
        raise NotImplementedError(
            "Only restricted references are currently supported."
        )

    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ, dtype=float)
    mo_energy = np.asarray(mf.mo_energy, dtype=float)

    if mo_coeff.ndim != 2:
        raise ValueError(
            "mf.mo_coeff must be a two-dimensional array."
        )

    nmo = mo_coeff.shape[1]

    if mo_occ.shape != (nmo,):
        raise ValueError(
            "mf.mo_occ must contain one occupation per MO."
        )

    if mo_energy.shape != (nmo,):
        raise ValueError(
            "mf.mo_energy must contain one energy per MO."
        )

    if not np.all(np.isin(mo_occ, (0.0, 2.0))):
        raise ValueError(
            "Only closed-shell restricted occupations of 0 or 2 "
            "are currently supported."
        )

    active = tuple(
        int(index)
        for index in active_orbitals
    )

    if not active:
        raise ValueError(
            "active_orbitals must contain at least one orbital."
        )

    if len(set(active)) != len(active):
        raise ValueError(
            "Active orbital indices must be unique."
        )

    if any(index < 0 or index >= nmo for index in active):
        raise IndexError(
            "An active orbital index is outside the MO range."
        )

    occupied = tuple(
        int(index)
        for index in np.flatnonzero(mo_occ == 2.0)
    )
    virtual = tuple(
        int(index)
        for index in np.flatnonzero(mo_occ == 0.0)
    )

    transitions = particle_hole_pairs(
        occupied,
        virtual,
    )

    residual_transitions = exclude_active_transitions(
        transitions,
        active,
    )

    bare_eri = bare_active_eri(
        mf,
        active,
    )

    nact = len(active)
    npairs = nact * nact

    # v[(p,q), (r,s)] = (pq|rs)
    bare_interaction = np.asarray(
        bare_eri,
        dtype=np.complex128,
    ).reshape(npairs, npairs)

    if not residual_transitions:
        return bare_interaction.reshape(
            nact,
            nact,
            nact,
            nact,
        )

    gaps = transition_energies(
        residual_transitions,
        mo_energy,
    )

    weights = polarization_weights(
        gaps,
        omega=omega,
        eta=eta,
    )


    # S[(p,q), I] = ∫ phi_p phi_q phi_i phi_a dr
    product_overlaps = product_overlap_matrix(
        mf,
        pair_orbitals=active,
        transitions=residual_transitions,
    )

    residual_polarizability = build_polarizability(
        product_overlaps,
        weights,
    )

    identity = np.eye(
        npairs,
        dtype=np.complex128,
    )

    dielectric = (
        identity
        - bare_interaction @ residual_polarizability
    )

    # W = (I - v P_r)^(-1) v
    screened_interaction = np.linalg.solve(
        dielectric,
        bare_interaction,
    )

    return screened_interaction.reshape(
        nact,
        nact,
        nact,
        nact,
    )