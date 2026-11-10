import pytest
import numpy as np
from src.data_loader import HSIDataLoader

@pytest.fixture
def synthetic_data():
    """Create synthetic MSI and HSI data."""
    msi = np.random.rand(100, 100, 3).astype(np.float32)
    hsi = np.random.rand(50, 50, 10).astype(np.float32)
    # Introduce some NaNs for testing preprocessing
    msi[10:20, 10:20, :] = np.nan
    hsi[5:10, 5:10, :] = np.nan
    return msi, hsi

def test_load_and_preprocess(synthetic_data, tmp_path):
    """Test HSIDataLoader's load_and_preprocess method."""
    msi, hsi = synthetic_data
    
    # Save synthetic data to temporary files (mock rasterio behavior)
    import rasterio
    msi_path = tmp_path / "msi.tif"
    hsi_path = tmp_path / "hsi.dat"
    
    # Mock rasterio write (simplified)
    with rasterio.open(msi_path, 'w', driver='GTiff', height=msi.shape[0], width=msi.shape[1], count=msi.shape[2], dtype='float32') as dst:
        for i in range(msi.shape[2]):
            dst.write(msi[:, :, i], i + 1)
    with rasterio.open(hsi_path, 'w', driver='GTiff', height=hsi.shape[0], width=hsi.shape[1], count=hsi.shape[2], dtype='float32') as dst:
        for i in range(hsi.shape[2]):
            dst.write(hsi[:, :, i], i + 1)
    
    loader = HSIDataLoader()
    msi_processed, hsi_processed = loader.load_and_preprocess(str(msi_path), str(hsi_path))
    
    # Check output shapes
    assert msi_processed.shape == (100, 100, 3)
    assert hsi_processed.shape == (50, 50, 10)
    
    # Check NaN handling
    assert not np.any(np.isnan(msi_processed))
    assert not np.any(np.isnan(hsi_processed))
    
    # Check data integrity (values should be similar to original non-NaN regions)
    assert np.allclose(msi_processed[0:10, 0:10, :], msi[0:10, 0:10, :], rtol=1e-5, atol=1e-5)

def test_preprocess_data_invalid_input():
    """Test preprocessing with invalid input dimensions."""
    loader = HSIDataLoader()
    invalid_data = np.random.rand(100, 100)  # 2D instead of 3D
    with pytest.raises(ValueError, match="Input data must be 3D"):
        loader.preprocess_data(invalid_data)

def test_load_image_accepts_gdal_virtual_paths():
    """GDAL virtual paths are not on disk, so they must not be rejected by the existence check."""
    import rasterio
    from rasterio.io import MemoryFile
    data = np.ones((1, 4, 4), dtype=np.float32)
    with MemoryFile() as mem:
        with mem.open(driver='GTiff', height=4, width=4, count=1, dtype='float32') as dst:
            dst.write(data)
        img = HSIDataLoader.load_image(mem.name)
    assert img.shape == (4, 4, 1)
