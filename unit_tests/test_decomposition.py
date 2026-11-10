import numpy as np
import pytest
from src.decomposition import Decomposition


@pytest.mark.parametrize("n_bands", [3, 12, 50])
def test_component_blocks_share_pixel_rows(n_bands):
    """All three decompositions return (patch pixels, n_components) so they can be hstacked."""
    patch = np.random.default_rng(0).random((12, 12, n_bands)).astype(np.float32)
    blocks = [Decomposition.wavelet_3d_transform(patch, 5),
              Decomposition.fastica_decomposition(patch, 5),
              Decomposition.nmf_decomposition(patch, 5)]
    for block in blocks:
        assert block.shape == (144, 5)
        assert np.all(np.isfinite(block))
    assert np.hstack(blocks).shape == (144, 15)


def test_wavelet_is_low_pass():
    """Zeroed detail coefficients leave a smoothed copy of the patch, not zeros."""
    patch = np.random.default_rng(1).random((12, 12, 8)).astype(np.float32)
    W = Decomposition.wavelet_3d_transform(patch, 5)
    assert np.all(np.linalg.norm(W, axis=0) > 0.99)
