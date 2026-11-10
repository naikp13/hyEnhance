import pytest
import numpy as np
from src.patch_processor import PatchProcessor

@pytest.fixture
def synthetic_patch_data():
    """Create synthetic MSI and HSI data for patch processing."""
    hsi = np.random.rand(50, 50, 10).astype(np.float32)
    msi = np.random.rand(100, 100, 3).astype(np.float32)
    return hsi, msi

def test_patch_processor_run_parallel(synthetic_patch_data):
    """Test PatchProcessor's run_parallel method."""
    hsi, msi = synthetic_patch_data
    processor = PatchProcessor(hsi, msi, n_components=5, n_atoms=5, lambda_reg=0.0005)
    
    # Run with small patch size and stride for testing
    hsi_hr = processor.run_parallel(patch_size=8, stride=4)
    
    # Check output shape
    assert hsi_hr.shape == (100, 100, 10)
    
    # Check that output contains non-zero values (indicating processing occurred)
    assert np.any(hsi_hr != 0)
    
    # Check that output is finite
    assert np.all(np.isfinite(hsi_hr))

def test_process_patch_invalid_data(synthetic_patch_data):
    """Test patch processing with invalid (all NaN) data."""
    hsi, msi = synthetic_patch_data
    processor = PatchProcessor(hsi, msi, n_components=5, n_atoms=5, lambda_reg=0.0005)
    
    # Create invalid patch data
    processor.hsi[0:8, 0:8, :] = np.nan
    processor.msi_lr = np.random.rand(50, 50, 3).astype(np.float32)
    processor.msi_lr[0:8, 0:8, :] = np.nan
    
    result = processor.process_patch(0, 0, patch_size=8)
    assert result is None  # Should return None for invalid patch

def test_run_parallel_raises_when_every_patch_fails(synthetic_patch_data, monkeypatch):
    """A stage that fails on every patch must not pass silently as an all-zero residual."""
    from src.patch_processor import PatchProcessingError
    hsi, msi = synthetic_patch_data
    processor = PatchProcessor(hsi, msi, n_components=5, n_atoms=5, lambda_reg=0.0005)

    def broken(*args, **kwargs):
        raise ValueError("boom")
    monkeypatch.setattr(processor.decomposition, "wavelet_3d_transform", broken)
    with pytest.raises(PatchProcessingError, match="boom"):
        processor.run_parallel(patch_size=8, stride=4)


def test_run_parallel_reports_stats(synthetic_patch_data):
    hsi, msi = synthetic_patch_data
    processor = PatchProcessor(hsi, msi, n_components=5, n_atoms=5, lambda_reg=0.0005)
    processor.run_parallel(patch_size=8, stride=8)
    assert processor.stats["failed"] == 0
    assert processor.stats["ok"] > 0
