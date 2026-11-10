import warnings

import numpy as np
import pywt
from sklearn.decomposition import FastICA, NMF
from sklearn.exceptions import ConvergenceWarning


def _normalize_columns(W):
    """Scale each column to unit L2 norm (all-zero columns stay zero)."""
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    return W / np.where(norms > 0, norms, 1.0)


def _pad_columns(W, n_components):
    """Zero-pad to n_components columns when fewer components could be extracted."""
    if W.shape[1] < n_components:
        W = np.pad(W, ((0, 0), (0, n_components - W.shape[1])), mode='constant')
    return W[:, :n_components]


class Decomposition:
    """Signal decompositions of an HSI patch.

    Every method takes a patch of shape (rows, cols, bands) and returns a
    per-pixel component matrix of shape (rows * cols, n_components), so the
    three blocks can be stacked side by side.
    """

    @staticmethod
    def wavelet_3d_transform(data, n_components):
        """Low-pass (approximation-only) reconstruction of a 3D wavelet transform."""
        with warnings.catch_warnings():
            # level 3 is deeper than a small patch allows; pywt warns but still works
            warnings.simplefilter("ignore", UserWarning)
            coeffs = pywt.wavedecn(data, 'db1', level=3)
            # Zero the detail coefficients instead of passing None, which newer pywt rejects
            details = [{k: np.zeros_like(v) for k, v in d.items()} for d in coeffs[1:]]
            approx = pywt.waverecn([coeffs[0]] + details, 'db1')
        approx = approx[:data.shape[0], :data.shape[1], :data.shape[2]]
        W = approx.reshape(-1, data.shape[-1])
        return _normalize_columns(_pad_columns(W, n_components))

    @staticmethod
    def fastica_decomposition(data, n_components):
        """FastICA sources per pixel: pixels are samples, bands are features."""
        data_2d = data.reshape(-1, data.shape[-1])
        k = min(n_components, data_2d.shape[1], data_2d.shape[0])
        transformer = FastICA(n_components=k, random_state=0, max_iter=200, whiten='unit-variance')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            S = transformer.fit_transform(data_2d)
        return _normalize_columns(_pad_columns(S, n_components))

    @staticmethod
    def nmf_decomposition(data, n_components):
        """NMF abundances per pixel: pixels are samples, bands are features."""
        data_2d = np.abs(data.reshape(-1, data.shape[-1]))
        k = min(n_components, data_2d.shape[1], data_2d.shape[0])
        transformer = NMF(n_components=k, init='random', random_state=0, max_iter=200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            W = transformer.fit_transform(data_2d)
        return _normalize_columns(_pad_columns(W, n_components))
