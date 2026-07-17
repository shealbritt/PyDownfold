from pyscf import gto, scf

from pydownfold.spaces import OrbitalSpace


def test_space_from_rhf() -> None:
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()

    space = OrbitalSpace.from_scf(mf, active_orbitals=[0, 1])

    assert space.nmo == 2
    assert space.active == (0, 1)
    assert space.occupied == (0,)
    assert space.virtual == (1,)
    assert space.active_occupied == (0,)
    assert space.active_virtual == (1,)