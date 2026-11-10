import numpy as np
from src.sparse_coding import SparseCoding


def test_dictionary_split_shapes():
    rng = np.random.default_rng(0)
    D_msi, D_comp = SparseCoding.train_dictionary(rng.standard_normal((144, 3)), rng.standard_normal((144, 15)), 5)
    assert D_msi.shape == (3, 5)
    assert D_comp.shape == (15, 5)


def test_residual_in_band_space():
    rng = np.random.default_rng(0)
    p, f, bands = 8, 2, 10
    residual = SparseCoding().sparse_code_residual(
        rng.random((p, p, 3)), rng.random((p * f, p * f, 3)), rng.random((p, p, 15)),
        n_atoms=5, f=f, lambda_reg=0.0005, hsi_patch=rng.random((p, p, bands)))
    assert residual.shape == (p * f, p * f, bands)
    assert np.all(np.isfinite(residual))
    assert np.any(residual != 0)


def test_band_map_without_ridge_handles_singular_components():
    rng = np.random.default_rng(0)
    comp = rng.random((144, 15))
    comp[:, 10:] = 0  # zero-padded components, as produced when fewer can be extracted
    M = SparseCoding.component_to_band_map(comp, rng.random((144, 12)), ridge=0)
    assert M.shape == (15, 12) and np.all(np.isfinite(M))
