import numpy as np
import cv2
from .data_loader import HSIDataLoader
from .patch_processor import PatchProcessor
from .upsampler import HSIUpsampler

class HSIEnhancer:
    """Main class for HSI resolution enhancement by MSI fusion."""
    
    def __init__(self, msi_path, hsi_path, n_components=5, n_atoms=5, lambda_reg=0.0005):
        self.loader = HSIDataLoader()
        self.msi, self.hsi = self.loader.load_and_preprocess(msi_path, hsi_path)
        self.n_components = n_components
        self.n_atoms = n_atoms
        self.lambda_reg = lambda_reg
        self.upsampler = HSIUpsampler()

    def fuse_to_enhance(self, patch_size=12, stride=1, guide_radius=1, detail_weight=3.5, residual_weight=1.0):
        """Perform HSI enhancement by fusing with MSI.

        residual_weight scales the patch-wise sparse residual before guided filtering
        (0 skips the patch stage entirely).
        """
        patch_processor = PatchProcessor(self.hsi, self.msi, self.n_components, self.n_atoms, self.lambda_reg)
        if residual_weight:
            hsi_hr = residual_weight * patch_processor.run_parallel(patch_size, stride)
        else:
            hsi_hr = np.zeros((self.msi.shape[0], self.msi.shape[1], self.hsi.shape[2]), dtype=np.float32)
        self.patch_stats = patch_processor.stats
    
        # Define default and fallback band indices
        selected_bands = [1, 7, 11]
        fallback_bands = [0, 1, 2]
    
        # Check if selected_bands are valid
        num_bands = self.msi.shape[-1]
        if any(band >= num_bands for band in selected_bands):
            selected_bands = fallback_bands[:num_bands]  # Use fallback, adjust if num_bands < 3
    
        msi_guide = self.msi[..., selected_bands].astype(np.float32)

        hsi_upsampled = self.upsampler.enhanced_hsi_upsampling(self.hsi, self.msi, msi_guide, detail_weight)

        for band in range(hsi_upsampled.shape[-1]):
            hsi_hr_band = cv2.ximgproc.guidedFilter(
                guide=msi_guide,
                src=hsi_upsampled[..., band] + hsi_hr[..., band],
                radius=guide_radius,
                eps=0.0001
            )
            hsi_upsampled[..., band] = hsi_hr_band
    
        return hsi_upsampled
