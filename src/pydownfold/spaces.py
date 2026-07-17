from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OrbitalSpace:
    """Restricted closed-shell SCF orbital space."""

    mo_coeff: FloatArray
    mo_energy: FloatArray
    mo_occ: FloatArray
    active: tuple[int, ...]

    def __post_init__(self) -> None:
        mo_coeff = np.asarray(self.mo_coeff, dtype=float)
        mo_energy = np.asarray(self.mo_energy, dtype=float)
        mo_occ = np.asarray(self.mo_occ, dtype=float)
        active = tuple(int(index) for index in self.active)

        if mo_coeff.ndim != 2:
            raise ValueError(
                "mo_coeff must be a two-dimensional array."
            )

        nmo = mo_coeff.shape[1]

        if mo_energy.shape != (nmo,):
            raise ValueError(
                "mo_energy must contain one value per MO."
            )

        if mo_occ.shape != (nmo,):
            raise ValueError(
                "mo_occ must contain one value per MO."
            )

        if len(set(active)) != len(active):
            raise ValueError(
                "Active orbital indices must be unique."
            )

        if any(index < 0 or index >= nmo for index in active):
            raise IndexError(
                "An active orbital index is outside the MO range."
            )

        if not np.all(np.isfinite(mo_coeff)):
            raise ValueError(
                "mo_coeff must contain only finite values."
            )

        if not np.all(np.isfinite(mo_energy)):
            raise ValueError(
                "mo_energy must contain only finite values."
            )

        if not np.all(np.isfinite(mo_occ)):
            raise ValueError(
                "mo_occ must contain only finite values."
            )

        if not np.all(np.isin(mo_occ, (0.0, 2.0))):
            raise ValueError(
                "Version 0.1 supports closed-shell occupations "
                "of 0 or 2 only."
            )

        object.__setattr__(self, "mo_coeff", mo_coeff)
        object.__setattr__(self, "mo_energy", mo_energy)
        object.__setattr__(self, "mo_occ", mo_occ)
        object.__setattr__(self, "active", active)

    @property
    def nmo(self) -> int:
        return self.mo_coeff.shape[1]

    @property
    def occupied(self) -> tuple[int, ...]:
        return tuple(
            int(index)
            for index in np.flatnonzero(self.mo_occ == 2.0)
        )

    @property
    def virtual(self) -> tuple[int, ...]:
        return tuple(
            int(index)
            for index in np.flatnonzero(self.mo_occ == 0.0)
        )

    @property
    def active_occupied(self) -> tuple[int, ...]:
        active = set(self.active)

        return tuple(
            index
            for index in self.occupied
            if index in active
        )

    @property
    def active_virtual(self) -> tuple[int, ...]:
        active = set(self.active)

        return tuple(
            index
            for index in self.virtual
            if index in active
        )

    @property
    def inactive_occupied(self) -> tuple[int, ...]:
        active = set(self.active)

        return tuple(
            index
            for index in self.occupied
            if index not in active
        )

    @property
    def external_virtual(self) -> tuple[int, ...]:
        active = set(self.active)

        return tuple(
            index
            for index in self.virtual
            if index not in active
        )

    @classmethod
    def from_scf(
        cls,
        mf,
        active_orbitals: Sequence[int],
    ) -> "OrbitalSpace":
        """Construct an orbital space from a converged RHF or RKS result."""
        if not getattr(mf, "converged", False):
            raise ValueError(
                "The supplied mean-field calculation is not converged."
            )

        if isinstance(mf.mo_coeff, tuple):
            raise NotImplementedError(
                "Version 0.1 supports restricted references only."
            )

        return cls(
            mo_coeff=np.asarray(mf.mo_coeff),
            mo_energy=np.asarray(mf.mo_energy),
            mo_occ=np.asarray(mf.mo_occ),
            active=tuple(int(index) for index in active_orbitals),
        )