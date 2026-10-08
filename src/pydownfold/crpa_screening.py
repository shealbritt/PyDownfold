from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pyscf import mcscf

from .crpa import kernel as crpa_kernel
from .integrals import bare_active_eri

@dataclass
class BareHamiltonian:
    h1: Array
    h2: Array
    ecore: float
    info: dict[str, Any]
    
Array = NDArray[np.float64]


@dataclass
class ScreenedHamiltonian:
    h1: Array
    h2: Array
    ecore: float
    h1_bare: Array
    h2_bare: Array
    t_dc: Array
    info: dict[str, Any]


def build_tdc_rhf(
    delta_h2: Array,
    density: Array,
) -> Array:
    """Build the RHF spatial-orbital double-counting correction."""
    coulomb = np.einsum(
        "rs,pqrs->pq",
        density,
        delta_h2,
        optimize=True,
    )

    exchange = np.einsum(
        "rs,prsq->pq",
        density,
        delta_h2,
        optimize=True,
    )

    t_dc = coulomb - 0.5 * exchange

    return 0.5 * (t_dc + t_dc.T)

def build_bare_hamiltonian(
    mf,
    active_orbitals: Sequence[int],
    nelecas,
) -> BareHamiltonian:
    """
    Build the bare RHF active-space Hamiltonian in the spatial-orbital
    convention used by PySCF CASCI.
    """

    active_orbitals = tuple(int(i) for i in active_orbitals)
    ncas = len(active_orbitals)

    cas = mcscf.CASCI(
        mf,
        ncas,
        nelecas,
    )

    expected_active = tuple(
        range(cas.ncore, cas.ncore + ncas)
    )

    if active_orbitals != expected_active:
        raise ValueError(
            "For now, active_orbitals must match PySCF's default "
            f"CASCI window {expected_active}."
        )

    h1, ecore = cas.get_h1eff(
        mf.mo_coeff
    )

    h1 = np.asarray(
        np.real(h1),
        dtype=float,
    )

    h2 = bare_active_eri(
        mf,
        active_orbitals,
    )

    h2 = np.asarray(
        np.real(h2),
        dtype=float,
    )

    return BareHamiltonian(
        h1=h1,
        h2=h2,
        ecore=float(ecore),
        info={
            "active_orbitals": active_orbitals,
            "ncas": ncas,
            "nelecas": nelecas,
            "ncore": cas.ncore,
        },
    )
def build_screened_hamiltonian(
    mf,
    active_orbitals: Sequence[int],
    nelecas,
    *,
    omega: float = 0.0,
    eta: float = 1e-8,
    apply_double_counting: bool = True,
) -> ScreenedHamiltonian:
    """
    Build a cRPA-screened RHF active-space Hamiltonian.
    """

    bare = build_bare_hamiltonian(
        mf,
        active_orbitals,
        nelecas,
    )

    active_orbitals = tuple(
        int(i) for i in active_orbitals
    )

    # ------------------------------------------------------------
    # Screened two-body interaction
    # ------------------------------------------------------------

    h2_screened = crpa_kernel(
        mf,
        active_orbitals,
        omega=omega,
        eta=eta,
    )

    max_imag = np.max(
        np.abs(np.imag(h2_screened))
    )

    if max_imag > 1e-8:
        raise ValueError(
            "Screened interaction has a non-negligible imaginary part: "
            f"max |Im(W)| = {max_imag:.3e}"
        )

    h2_screened = np.asarray(
        np.real(h2_screened),
        dtype=float,
    )

    # ------------------------------------------------------------
    # Change in the active-space two-body interaction
    # ------------------------------------------------------------

    delta_h2 = (
        h2_screened
        - bare.h2
    )

    # ------------------------------------------------------------
    # RHF active-space reference density
    # ------------------------------------------------------------

    active_density = np.diag(
        np.asarray(mf.mo_occ)[
            list(active_orbitals)
        ]
    )

    # ------------------------------------------------------------
    # Double-counting correction
    # ------------------------------------------------------------

    if apply_double_counting:
        t_dc = build_tdc_rhf(
            delta_h2,
            active_density,
        )
    else:
        t_dc = np.zeros_like(
            bare.h1
        )

    # ------------------------------------------------------------
    # Screened one-body term
    # ------------------------------------------------------------

    h1_screened = (
        bare.h1
        - t_dc
    )

    info = {
        **bare.info,
        "delta_h2_norm": np.linalg.norm(delta_h2),
        "tdc_norm": np.linalg.norm(t_dc),
        "apply_double_counting": apply_double_counting,
        "omega": omega,
        "eta": eta,
    }

    return ScreenedHamiltonian(
        h1=np.asarray(h1_screened),
        h2=np.asarray(h2_screened),
        ecore=bare.ecore,
        h1_bare=np.asarray(bare.h1),
        h2_bare=np.asarray(bare.h2),
        t_dc=np.asarray(t_dc),
        info=info,
    )