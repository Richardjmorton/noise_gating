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


class ImageCube:
    """
    Class for Image sub cubes
    """
    def __init__(self, image, fourier_image):
        self.image = image
        self.fourier_image = fourier_image
        self.betas = None

    def _estimate_shot_noise(self):
        fac = np.sum(np.sqrt(self.image[self.image > 0]))
        betas = np.abs(self.fourier_image)/fac
        self.betas = betas
        return betas

    def _image_indepen_noise(self):
        betas = np.abs(self.fourier_image)
        self.betas = betas
        return betas

    def gate_filter(self, beta, gamma=1.5):
        fourier_amp = np.abs(self.fourier_image)
        threshold = gamma * beta
        filt = np.logical_not(fourier_amp < threshold)
        return filt

    def wiener_filter(self, beta, gamma=1.5):
        filt = np.abs(self.fourier_image) / \
            (gamma*beta+np.abs(self.fourier_image))
        return filt
