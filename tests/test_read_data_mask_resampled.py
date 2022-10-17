from brainage import binarize_3d
from nibabel import Nifti1Image
import numpy as np

def _make_image():
    return Nifti1Image(
        np.random.default_rng(seed=5).integers(low=0, high=5, size=(5, 5, 2)),
        np.eye(4),
    )

def test_binarize_3d():
    img = _make_image()
    bin_img = binarize_3d(img, threshold=2)
    
    assert np.min(bin_img.get_fdata()) == 0
    assert np.max(bin_img.get_fdata()) == 1
    np.testing.assert_array_equal(np.unique(bin_img.get_fdata()),np.array([0, 1]))
