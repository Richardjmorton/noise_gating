import numpy as np
from itertools import product
from noise_gate.utils import do_apod3d, ImageCube


__all__ = ['noise_gate']


def calc_noise_profile(data, win_size=12, perc=50, beta_const=False,
                       image_indepen_noise=False):
    """
   Calculates the noise profile for filtering the data cube.

    Parameters
    ----------
    data : ndarray
     Data cube - expects order (x, y, t)
    win_size : int
     length of sub-cube side
    perc: int
     percentage value for noise estimate
     default 50 (%) (median)
    beta_const: boolean
     Set for single value of beta across Fourier space
    image_indepen_noise: boolean
     Select for calculation of image independent noise
    gate: boolean

    """
    nx, ny, nz = data.shape

    half_win = win_size // 2

    # Apodisation cube for sub-images
    apod = do_apod3d(win_size, win_size, win_size)

    # Loop for calculating noise profile -> Is this what can be parallelised?
    # No, this is quite quick

    print("Calculating noise model...")

    strides_x = np.arange(half_win, nx, step=win_size)
    strides_y = np.arange(half_win, ny, step=win_size)

    noise_elem = len(strides_x)*len(strides_y)

    # Scratch array
    noise_arr = np.zeros(
        shape=(win_size, win_size, win_size, noise_elem), dtype='float64')

    # get estimates for noise from multiple sub-images
    fft = np.fft.fftn
    for count, (i, j) in enumerate(product(strides_x, strides_y)):

        # define sub-image coordinates
        x = (int(i-half_win), int(i+half_win))
        y = (int(j-half_win), int(j+half_win))

        # move on if coordinates out of bounds
        if y[1] > ny or x[1] > nx:
            continue

        # define sub-image and apodise
        sub_image = data[x[0]:x[1], y[0]:y[1], 0:win_size] * apod
        fourier_image = fft(sub_image)

        subImageCube = ImageCube(sub_image, fourier_image)

        if image_indepen_noise:
            subImageCube._image_indepen_noise()
        else:
            subImageCube._estimate_shot_noise()

        noise_arr[:, :, :, count] = subImageCube.betas

    # Calculate noise profile
    if beta_const:
        betas = np.percentile(noise_arr.ravel(), perc)
    else:
        betas = np.percentile(noise_arr, perc, axis=3)

    return betas


def run_filter(data, betas, win_size=12, beta_const=False,
               image_indepen_noise=False, gate=False):

    """
    Filters the data given the noise profile.

    Parameters
    ----------
    data : ndarray
     Data cube - expects order (x, y, t)
    betas: ndarray
     noise profile
    win_size : int
     length of sub-cube side
    perc: int
     percentage value for noise estimate
     default 50 (%) (median)
    beta_const: boolean
     Set for single value of beta across Fourier space
    image_indepen_noise: boolean
     Select for calculation of image independent noise
    gate: boolean

    """

    nx, ny, nz = data.shape
    half_win = win_size // 2

    # Apodisation cube for sub-images
    apod = do_apod3d(win_size, win_size, win_size)

    # short-hand for fft functions
    fft = np.fft.fftn
    ifft = np.fft.ifftn
    fftshift = np.fft.fftshift

    over_sample_width = win_size/4  # using sin**4 window

    strides_x = np.arange(half_win, nx-half_win, step=over_sample_width)
    strides_y = np.arange(half_win, ny-half_win, step=over_sample_width)
    strides_z = np.arange(half_win, nz-half_win, step=over_sample_width)

    n_calc = len(strides_x)*len(strides_y)*len(strides_z)

    print('Implementing noise gating...')
    gated_data = np.zeros(shape=(nx, ny, nz), dtype='float64')  # scratch array
    for count, (i, j, k) in enumerate(product(strides_x,
                                              strides_y,
                                              strides_z)):

        # define sub-image coordinates
        x = (int(i-half_win), int(i+half_win))
        y = (int(j-half_win), int(j+half_win))
        z = (int(k-half_win), int(k+half_win))

        if y[1] > ny or x[1] > nx or z[1] > nz:
            continue

        sub_image = data[x[0]:x[1], y[0]:y[1], z[0]:z[1]] * apod
        fourier_image = fft(sub_image)

        subImageCube = ImageCube(sub_image, fourier_image)

        if image_indepen_noise:
            noise_profile = betas
        else:
            im_clip = np.copy(sub_image)
            im_clip[im_clip < 0] = 0
            im_total = np.sum(np.sqrt(im_clip))
            noise_profile = betas*im_total

        if gate:
            filt = subImageCube.gate_filter(noise_profile)
        else:
            filt = subImageCube.wiener_filter(noise_profile)

        # Keeping core region of DFT when using varying beta
        if not beta_const:
            core = (half_win-1, half_win+2)
            filt = fftshift(filt)

            filt[core[0]:core[1],
                 core[0]:core[1],
                 core[0]:core[1]] = 1

            filt = fftshift(filt)

        inverse_ft = (ifft(fourier_image*filt)*apod).real
        gated_data[x[0]:x[1], y[0]:y[1], z[0]:z[1]] += inverse_ft

        print('{:3.1f} % complete'.format(100*count/n_calc), end="\r")

    # correction factor for windowing
    # only applicable for sin^4 window
    gated_data /= (1.5)**3

    return gated_data


def noise_gate(data, win_size=12, beta_const=False,
               image_indepen_noise=False, gate=False, perc=50):
    """
    Noise-gate a data cube.

    Parameters
    ----------
    data : ndarray
     Data cube - expects order (x, y, t)
    win_size : int
     length of sub-cube side
    perc: int
     percentage value for noise estimate
     default 50 (%) (median)
    beta_const: boolean
     Set for single value of beta across Fourier space
     default false
    image_indepen_noise: boolean
     Select for calculation of image independent noise
     default false
    gate: boolean
     selects filter type. Either Weiner filter (False) or gating (True)
     default false
    """

    betas = calc_noise_profile(data, win_size=win_size, perc=perc,
                               beta_const=beta_const,
                               image_indepen_noise=image_indepen_noise)

    return run_filter(data, betas, win_size=win_size, beta_const=beta_const,
                      image_indepen_noise=image_indepen_noise, gate=gate)
