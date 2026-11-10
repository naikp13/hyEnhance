import numpy as np
import numba as nb
from scipy.ndimage import zoom


@nb.njit
def _fista(X, D, lambda_reg, max_iter, tol):
    """FISTA for min_a 0.5 * ||X - a D^T||^2 + lambda * ||a||_1, one row of `a` per sample.

    X: (n_samples, n_features), D: (n_features, n_atoms). Returns (n_samples, n_atoms).
    """
    n_samples = X.shape[0]
    n_atoms = D.shape[1]
    L = np.linalg.norm(D, 2) ** 2
    alpha = np.zeros((n_samples, n_atoms))
    if L == 0.0:
        return alpha
    y = alpha.copy()
    t = 1.0
    thresh = lambda_reg / L
    for k in range(max_iter):
        alpha_prev = alpha.copy()
        z = y + np.dot(X - np.dot(y, D.T), D) / L
        alpha = np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = alpha + ((t - 1.0) / t_next) * (alpha - alpha_prev)
        t = t_next
        if k > 5 and np.max(np.abs(alpha - alpha_prev)) < tol:
            break
    return alpha


def _standardize(a, mean, std):
    return (a - mean) / std


class SparseCoding:
    """Coupled-dictionary sparse coding of the patch-wise residual (P2SR).

    For one patch, a joint dictionary is learned at low resolution over
    [degraded MSI | HSI components]. Its MSI rows code the MSI at low and at
    high resolution, its component rows turn those codes into component
    estimates, and the high-frequency difference between the two estimates is
    mapped back to the HSI bands.
    """

    @staticmethod
    def fista(X, D, lambda_reg, max_iter=75, tol=1e-6):
        """Sparse codes of X (n_samples, n_features) over D (n_features, n_atoms)."""
        return _fista(np.ascontiguousarray(X, dtype=np.float64),
                      np.ascontiguousarray(D, dtype=np.float64),
                      float(lambda_reg), int(max_iter), float(tol))

    @staticmethod
    def train_dictionary(msi_lr_flat, hsi_comp_flat, n_atoms, alpha=1.0, n_iter=20):
        """Learn a joint dictionary over pixels described by [MSI features | component features].

        msi_lr_flat: (n_pixels, n_msi), hsi_comp_flat: (n_pixels, n_comp), both standardized.
        Minimizes 0.5 * ||X - A V||^2 + alpha * ||A||_1 with unit-norm atoms (rows of V), the same
        objective as sklearn's DictionaryLearning, by alternating FISTA sparse coding with a
        least-squares (MOD) dictionary update. This is much faster than sklearn on many small
        patches.
        Returns (D_msi, D_comp): (n_msi, n_atoms) and (n_comp, n_atoms), the two row blocks of
        the joint dictionary.
        """
        X = np.hstack([msi_lr_flat, hsi_comp_flat]).astype(np.float64)   # (n_pixels, n_features)
        n_features = X.shape[1]
        # SVD initialization, as in sklearn
        _, sv, Vt = np.linalg.svd(X, full_matrices=False)
        V = np.zeros((n_atoms, n_features))
        k = min(n_atoms, Vt.shape[0])
        V[:k] = Vt[:k]
        if n_atoms > k:  # more atoms than features: fill with seeded random unit vectors
            V[k:] = np.random.default_rng(0).standard_normal((n_atoms - k, n_features))
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        for _ in range(n_iter):
            A = _fista(X, np.ascontiguousarray(V.T), alpha, 75, 1e-6)
            used = np.abs(A).sum(0) > 0
            if not used.any():
                break
            V_new, *_ = np.linalg.lstsq(A[:, used], X, rcond=None)
            V[used] = V_new
            V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        D = V.T                                                          # (n_features, n_atoms)
        n_msi = msi_lr_flat.shape[1]
        return D[:n_msi], D[n_msi:]

    @staticmethod
    def component_to_band_map(hsi_comp_flat, hsi_flat, ridge=1.0):
        """Ridge-regression map M with (hsi - mean) ~ components @ M, shape (n_comp, n_bands).

        The wavelet, ICA and NMF blocks are strongly collinear, so a plain least-squares map can
        amplify the component residual. `ridge` is relative to the mean eigenvalue of the
        component Gram matrix (0 gives ordinary least squares).
        """
        centered = hsi_flat - hsi_flat.mean(axis=0, keepdims=True)
        G = hsi_comp_flat.T @ hsi_comp_flat
        lam = ridge * np.trace(G) / G.shape[0]
        return np.linalg.solve(G + lam * np.eye(G.shape[0]), hsi_comp_flat.T @ centered)

    def sparse_code_residual(self, msi_lr_patch, msi_hr_patch, hsi_components, n_atoms, f, lambda_reg,
                             hsi_patch=None, map_ridge=1.0):
        """High-resolution residual for one patch.

        msi_lr_patch: (p, p, n_msi) degraded MSI on the HSI grid
        msi_hr_patch: (p*f, p*f, n_msi) MSI at full resolution
        hsi_components: (p, p, n_comp) stacked wavelet / ICA / NMF components of the HSI patch
        hsi_patch: (p, p, n_bands) the HSI patch. If given, the residual is returned in HSI
            bands (p*f, p*f, n_bands); otherwise in component space (p*f, p*f, n_comp).
        map_ridge: regularization of the component-to-band map (see component_to_band_map).
        """
        p_lr = msi_lr_patch.shape[:2]
        p_hr = msi_hr_patch.shape[:2]
        msi_lr = msi_lr_patch.reshape(-1, msi_lr_patch.shape[-1]).astype(np.float64)
        msi_hr = msi_hr_patch.reshape(-1, msi_hr_patch.shape[-1]).astype(np.float64)
        comp = hsi_components.reshape(-1, hsi_components.shape[-1]).astype(np.float64)

        # Standardize each feature with low-resolution statistics so MSI digital numbers and
        # unit-norm components carry comparable weight in the joint dictionary
        m_mean, m_std = msi_lr.mean(0), msi_lr.std(0) + 1e-8
        c_mean, c_std = comp.mean(0), comp.std(0) + 1e-8
        msi_lr_s = _standardize(msi_lr, m_mean, m_std)
        msi_hr_s = _standardize(msi_hr, m_mean, m_std)
        comp_s = _standardize(comp, c_mean, c_std)

        D_msi, D_comp = self.train_dictionary(msi_lr_s, comp_s, n_atoms)

        # Code the MSI against the MSI rows only, then reconstruct components from the component rows
        a_hr = self.fista(msi_hr_s, D_msi, lambda_reg)
        a_lr = self.fista(msi_lr_s, D_msi, lambda_reg)
        comp_hr = (a_hr @ D_comp.T) * c_std + c_mean               # (p_hr pixels, n_comp)
        comp_lr = (a_lr @ D_comp.T) * c_std + c_mean               # (p_lr pixels, n_comp)

        # Residual = high-resolution estimate minus the upsampled low-resolution estimate,
        # so only the detail the MSI adds is kept and the dictionary's approximation bias cancels
        comp_hr = comp_hr.reshape(p_hr[0], p_hr[1], -1)
        comp_lr_up = zoom(comp_lr.reshape(p_lr[0], p_lr[1], -1),
                          (p_hr[0] / p_lr[0], p_hr[1] / p_lr[1], 1), order=3, mode='nearest')
        residual_comp = comp_hr - comp_lr_up

        if hsi_patch is None:
            return residual_comp.astype(np.float32)

        M = self.component_to_band_map(comp, hsi_patch.reshape(-1, hsi_patch.shape[-1]).astype(np.float64),
                                       ridge=map_ridge)
        residual = residual_comp.reshape(-1, residual_comp.shape[-1]) @ M
        return residual.reshape(p_hr[0], p_hr[1], -1).astype(np.float32)
