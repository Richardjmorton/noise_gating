import numpy as np


def do_apod3d(nx, ny, nz):
    '''
    Construct a 3D dimensional apodisation function

    Parameters
    ----------
        Dimensions of required cube
        nx - int
        ny - int
        nz - int

    '''

    apodx = np.sin(np.pi*(np.arange(0, nx, 1, dtype=float)+0.5)/nx) ** 2
    apody = np.sin(np.pi*(np.arange(0, ny, 1, dtype=float)+0.5)/ny) ** 2
    apodz = np.sin(np.pi*(np.arange(0, nz, 1, dtype=float)+0.5)/nz) ** 2

    apodxy = np.matmul(apodx[:, np.newaxis], apody[np.newaxis, :])
    apodxyz = np.matmul(apodxy[:, :, np.newaxis], apodz[np.newaxis, :])

    return apodxyz
