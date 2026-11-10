import logging
import os
import numpy as np
import rasterio
from scipy.ndimage import median_filter

# Set up logging
logging.basicConfig(level=logging.INFO, filename='processing.log', filemode='w')

class HSIDataLoader:
    """Handles loading and initial processing of HSI and MSI data."""
    
    @staticmethod
    def load_image(file_path):
        """Load image using rasterio and convert to float32."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such image file: {file_path}")
        with rasterio.open(file_path) as src:
            img = src.read().astype(np.float32)
            img = np.moveaxis(img, 0, -1)
            img[img <= 0] = np.nan
        return img

    @staticmethod
    def preprocess_data(data, window_size=3, nan_threshold=0.5):
        """Preprocess data by handling NaNs and applying median filter."""
        if data.ndim != 3:
            raise ValueError("Input data must be 3D (height, width, channels)")
        valid_mask = ~np.isnan(data)
        if np.any(valid_mask):
            data_filled = np.where(valid_mask, data, median_filter(data, size=window_size, mode='nearest'))
            data_filled = np.nan_to_num(data_filled, nan=np.nanmean(data_filled))
        else:
            data_filled = np.nan_to_num(data, nan=0)
        return data_filled

    def load_and_preprocess(self, msi_path, hsi_path, nan_threshold=0.5):
        """Load and preprocess MSI and HSI images."""
        msi = self.load_image(msi_path)
        hsi = self.load_image(hsi_path)

        # Filter out invalid HSI bands
        valid_bands = np.isnan(hsi).mean(axis=(0, 1)) <= nan_threshold
        eliminated_bands = np.where(~valid_bands)[0]
        if len(eliminated_bands) > 0:
            logging.info(f"Eliminated band numbers: {eliminated_bands}")
        else:
            logging.info("No bands were eliminated")
        hsi = hsi[..., valid_bands]

        # Preprocess images
        msi = self.preprocess_data(msi)
        hsi = self.preprocess_data(hsi)

        return msi, hsi
