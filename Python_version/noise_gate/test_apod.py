import pytest
import do_apod3d


def test_matmul(self):
    apod = do_apod3d(20, 20, 20)
    assert apod.shape == (20, 20, 20), 'Dimensions incorrect'

    apod = do_apod3d(20, 20, 10)
    assert apod.shape == (20, 20, 10), 'Dimensions incorrect'

    apod = do_apod3d(10, 20, 10)
    assert apod.shape == (10, 20, 10), 'Dimensions incorrect'

    apod = do_apod3d(10, 10, 20)
    assert apod.shape == (10, 10, 20), 'Dimensions incorrect'
