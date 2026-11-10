import logging
from collections import namedtuple

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import zoom
from tqdm import tqdm

from .decomposition import Decomposition
from .sparse_coding import SparseCoding

# Returned by process_patch when a patch raised, so run_parallel can count failures
PatchFailure = namedtuple("PatchFailure", ["x", "y", "error"])


class PatchProcessingError(RuntimeError):
    """Raised when the patch-wise residual stage fails on every patch."""


class PatchProcessor:
    """Handles patch-based processing for HSI enhancement."""

    def __init__(self, hsi, msi, n_components=5, n_atoms=5, lambda_reg=0.0005, n_jobs=4):
        self.hsi = hsi
        self.msi = msi
        self.n_components = n_components
        self.n_atoms = n_atoms
        self.lambda_reg = lambda_reg
        self.n_jobs = n_jobs
        self.decomposition = Decomposition()
        self.sparse_coding = SparseCoding()
        self.f = max(1, self.msi.shape[0] // self.hsi.shape[0])
        self.msi_lr = zoom(self.msi, (1 / self.f, 1 / self.f, 1), order=2, mode='nearest')
        self.stats = {}

    def process_patch(self, x, y, patch_size):
        """Process one patch.

        Returns (x_hr, y_hr, residual) on success, None when the patch has too few valid
        pixels, and a PatchFailure when processing raised.
        """
        try:
            hsi_patch = self.hsi[x:x + patch_size, y:y + patch_size, :]
            msi_patchLR = self.msi_lr[x:x + patch_size, y:y + patch_size, :]
            valid = np.isfinite(hsi_patch).all(axis=-1) & np.isfinite(msi_patchLR).all(axis=-1)
            if np.sum(valid) <= 5:
                return None

            msi_patchHR = self.msi[x * self.f:x * self.f + patch_size * self.f,
                                   y * self.f:y * self.f + patch_size * self.f, :]
            hsi_patch_clean = np.nan_to_num(hsi_patch, nan=np.nanmean(hsi_patch[valid]))
            msi_patchLR_clean = np.nan_to_num(msi_patchLR, nan=np.nanmean(msi_patchLR[valid]))
            msi_patchHR_clean = np.nan_to_num(msi_patchHR, nan=np.nanmean(msi_patchHR))

            wt_components = self.decomposition.wavelet_3d_transform(hsi_patch_clean, self.n_components)
            ica_components = self.decomposition.fastica_decomposition(hsi_patch_clean, self.n_components)
            nmf_components = self.decomposition.nmf_decomposition(hsi_patch_clean, self.n_components)
            combined_components = np.hstack([wt_components, ica_components, nmf_components])
            combined_components_2d = combined_components.reshape(patch_size, patch_size, -1)

            residual = self.sparse_coding.sparse_code_residual(
                msi_patchLR_clean, msi_patchHR_clean, combined_components_2d,
                self.n_atoms, self.f, self.lambda_reg, hsi_patch=hsi_patch_clean)
            if not np.all(np.isfinite(residual)):
                raise FloatingPointError("non-finite residual")
            return (x * self.f, y * self.f, residual)
        except Exception as e:
            logging.error(f"Patch ({x}, {y}) failed: {e}")
            return PatchFailure(x, y, f"{type(e).__name__}: {e}")

    def run_parallel(self, patch_size=12, stride=1):
        """Run patch processing in parallel and blend overlapping residuals.

        Raises PatchProcessingError if every processable patch failed. Counts are kept in
        `self.stats` (ok, skipped, failed, first_error).
        """
        self.f = max(1, self.msi.shape[0] // self.hsi.shape[0])
        self.msi_lr = zoom(self.msi, (1 / self.f, 1 / self.f, 1), order=2, mode='nearest')

        coords = [(x, y) for x in range(0, self.hsi.shape[0] - patch_size + 1, stride)
                  for y in range(0, self.hsi.shape[1] - patch_size + 1, stride)]

        hsi_hr = np.zeros((self.msi.shape[0], self.msi.shape[1], self.hsi.shape[2]), dtype=np.float32)
        counts = np.zeros((self.msi.shape[0], self.msi.shape[1]), dtype=np.int32)

        # Processes, not threads: sklearn's FastICA/NMF hold the GIL, so threads ran ~5x slower
        backend = 'threading' if len(coords) < 50 else 'loky'
        results = Parallel(n_jobs=self.n_jobs, backend=backend)(
            delayed(self.process_patch)(x, y, patch_size) for x, y in tqdm(coords, desc="Processing patches")
        )

        failures = [r for r in results if isinstance(r, PatchFailure)]
        n_skipped = sum(r is None for r in results)
        n_ok = len(results) - len(failures) - n_skipped
        self.stats = dict(ok=n_ok, skipped=n_skipped, failed=len(failures),
                          first_error=failures[0].error if failures else None)
        if failures:
            logging.warning(f"{len(failures)} of {len(results)} patches failed; first error: {failures[0].error}")
        if n_ok == 0 and failures:
            raise PatchProcessingError(
                f"All {len(failures)} processable patches failed; first error at patch "
                f"({failures[0].x}, {failures[0].y}): {failures[0].error}")

        for result in results:
            if result is None or isinstance(result, PatchFailure):
                continue
            x_start, y_start, residual = result
            x_end = x_start + residual.shape[0]
            y_end = y_start + residual.shape[1]
            x_start = min(x_start, self.msi.shape[0] - (x_end - x_start))
            y_start = min(y_start, self.msi.shape[1] - (y_end - y_start))
            x_end = min(x_end, self.msi.shape[0])
            y_end = min(y_end, self.msi.shape[1])
            n_channels = min(residual.shape[-1], hsi_hr.shape[-1])
            hsi_hr[x_start:x_end, y_start:y_end, :n_channels] += residual[..., :n_channels]
            counts[x_start:x_end, y_start:y_end] += 1

        valid_mask = counts > 0
        hsi_hr[valid_mask] = hsi_hr[valid_mask] / counts[valid_mask, np.newaxis]

        return hsi_hr
