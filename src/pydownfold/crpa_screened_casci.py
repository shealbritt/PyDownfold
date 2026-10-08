from __future__ import annotations

import numpy as np
from pyscf import fci, mcscf

from .crpa_screening import ScreenedHamiltonian


def run_screened_casci(
    mf,
    nelecas,
    hamiltonian: ScreenedHamiltonian,
    *,
    nroots: int = 1,
):
    """Run PySCF CASCI using a prebuilt screened Hamiltonian."""

    ncas = hamiltonian.h1.shape[0]

    cas = mcscf.CASCI(
        mf,
        ncas,
        nelecas,
    )

    cas.fcisolver = fci.direct_spin1.FCI(mf.mol)
    cas.fcisolver.nroots = nroots

    h1 = np.array(
        hamiltonian.h1,
        copy=True,
    )

    h2 = np.array(
        hamiltonian.h2,
        copy=True,
    )

    ecore = float(
        hamiltonian.ecore
    )

    def get_h1eff(
        mo_coeff=None,
        *args,
        **kwargs,
    ):
        return h1, ecore

    def get_h2eff(
        mo_coeff=None,
        *args,
        **kwargs,
    ):
        return h2

    cas.get_h1eff = get_h1eff
    cas.get_h2eff = get_h2eff

    cas.kernel(mf.mo_coeff)

    energies = np.atleast_1d(
        np.asarray(
            cas.e_tot,
            dtype=float,
        )
    )

    return energies, cas