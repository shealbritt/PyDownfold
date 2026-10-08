from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Transition = tuple[int, int]


def particle_hole_pairs(
    occupied: Sequence[int],
    virtuals: Sequence[int],
) -> tuple[Transition, ...]:
    """Return all occupied-to-virtual orbital pairs."""
    return tuple(
        (int(i), int(a))
        for i in occupied
        for a in virtuals
    )

def exclude_active_transitions(
    transitions: Sequence[Transition],
    active_orbitals: Sequence[int],
) -> tuple[Transition, ...]:
    """Exclude transitions wholly contained in the active space."""
    active = set(int(index) for index in active_orbitals)

    return tuple(
        (i, a)
        for i, a in transitions
        if not (i in active and a in active)
    )

def transition_energies(
    transitions: Sequence[Transition],
    mo_energies: FloatArray,
) -> FloatArray:
    """Return epsilon_a - epsilon_i for each particle-hole transition."""
    mo_energies = np.asarray(mo_energies, dtype=float)
    transitions = tuple(transitions)

    delta = np.asarray(
        [mo_energies[a] - mo_energies[i] for i, a in transitions],
        dtype=float,
    )

    if np.any(delta <= 0.0):
        raise ValueError(
            "Occupied-to-virtual transition energies must be positive."
        )

    return delta


def polarization_weights(
    transition_energies: FloatArray,
    omega: float = 0.0,
    eta: float = 1e-5,
    closed_shell: bool = True,
) -> ComplexArray:
    """Return frequency-dependent independent-particle response weights."""
    transition_energies = np.asarray(transition_energies, dtype=float)

    if transition_energies.ndim != 1:
        raise ValueError("transition_energies must be one-dimensional.")

    if eta <= 0.0:
        raise ValueError("eta must be positive.")

    weights = (
        1.0 / (omega - transition_energies + 1j * eta)
        - 1.0 / (omega + transition_energies + 1j * eta)
    )

    if closed_shell:
        weights = 2.0 * weights
    
    return weights

def build_polarizability(
    product_overlaps: FloatArray,
    weights: ComplexArray,
) -> ComplexArray:
    """Build P = S diag(weights) S.T."""
    product_overlaps = np.asarray(
        product_overlaps,
        dtype=float,
    )
    weights = np.asarray(
        weights,
        dtype=np.complex128,
    )

    if product_overlaps.ndim != 2:
        raise ValueError(
            "product_overlaps must have shape "
            "(npairs, ntransitions)."
        )

    if weights.ndim != 1:
        raise ValueError(
            "weights must be one-dimensional."
        )

    if product_overlaps.shape[1] != weights.shape[0]:
        raise ValueError(
            "weights must contain one value per transition."
        )

    return (
        product_overlaps * weights[None, :]
    ) @ product_overlaps.T 