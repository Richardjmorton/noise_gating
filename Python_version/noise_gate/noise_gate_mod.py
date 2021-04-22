import numpy as np
from numpy.fft import fftn, ifftn, fftshift
from itertools import product
from .utils.utils import do_apod3d, ImageCube

try:
    import cupy
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


__all__ = ['noise_gate']


def gate_filter(fourier_image, beta, gamma=1.5):
    fourier_amp = np.abs(fourier_image)
    threshold = gamma * beta
    filt = np.logical_not(fourier_amp < threshold)
    return filt


def wiener_filter(fourier_image, beta, gamma=1.5):
    filt = np.abs(fourier_image) / \
        (gamma*beta+np.abs(fourier_image))
    return filt


def calc_noise_profile(data, perc=50, win_size=12, beta_const=False,
                       image_indepen_noise=False):
    """
   Calculates the noise profile for filtering the data cube.

    Parameters
    ----------
    data : ndarray
     Data cube - expects order (t, y, x)
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
    nz, ny, nx = data.shape

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
    for count, (i, j) in enumerate(product(strides_x, strides_y)):

        # define sub-image coordinates
        x = (int(i-half_win), int(i+half_win))
        y = (int(j-half_win), int(j+half_win))

        # move on if coordinates out of bounds
        if y[1] > ny or x[1] > nx:
            continue

        # define sub-image and apodise
        sub_image = data[0:win_size, y[0]:y[1], x[0]:x[1]] * apod
        fourier_image = fftn(sub_image)

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


def get_filter(chunk_arr, fourier_image, betas, half_win, beta_const=False,
               image_indepen_noise=False, gate=False, gamma=1.5):

    if image_indepen_noise:
        noise_profile = betas
    else:
        im_clip = chunk_arr.copy()
        im_clip[im_clip < 0] = 0
        im_total = np.sum(np.sqrt(im_clip), axis=(1, 2, 3))
        noise_profile = betas*im_total[:, np.newaxis, np.newaxis,
                                       np.newaxis]

    if gate:
        filt = gate_filter(fourier_image, noise_profile, gamma=gamma)
    else:
        filt = wiener_filter(fourier_image, noise_profile, gamma=gamma)

    # Keeping core region of DFT when using varying beta
    if not beta_const:
        core = (half_win-1, half_win+2)
        filt = fftshift(filt, axes=(1, 2, 3))

        filt[:, core[0]:core[1],
             core[0]:core[1],
             core[0]:core[1]] = 1

        filt = fftshift(filt, axes=(1, 2, 3))
    return filt


def run_filter(data, betas, win_size=12, beta_const=False,
               image_indepen_noise=False, gate=False, gamma=1.5, use_gpu=True):

    """
    Filters the data given the noise profile.

    Parameters
    ----------
    data : ndarray
     Data cube - expects order (t, y, x)
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

    filter_kwargs = {'beta_const': beta_const,
                     'image_indepen_noise': image_indepen_noise,
                     'gate': gate,
                     'gamma': gamma}

    nz, ny, nx = data.shape
    half_win = win_size // 2

    # Apodisation cube for sub-images
    apod = do_apod3d(win_size, win_size, win_size)

    over_sample_width = win_size/4  # using sin**4 window

    strides_x = np.arange(half_win, nx-half_win, step=over_sample_width)
    strides_y = np.arange(half_win, ny-half_win, step=over_sample_width)
    strides_z = np.arange(half_win, nz-half_win, step=over_sample_width)

    n_calc = len(strides_x)*len(strides_y)

    print('Implementing noise gating...')
    gated_data = np.zeros(shape=(nz, ny, nx), dtype='float64')  # scratch array

    for count, (i, j) in enumerate(product(strides_x,
                                           strides_y)):

        # in this loop we process sub-images at a single location for
        # the entire data array. Much faster than doing the calculations one
        # at a time. Will be useful for GPU processing.

        # define spatial sub-image coordinates
        x = (int(i-half_win), int(i+half_win))
        y = (int(j-half_win), int(j+half_win))

        if y[1] > ny or x[1] > nx:
            continue

        # get all sub-images in z direction
        chunk = []
        for k in strides_z:
            z = (int(k-half_win), int(k+half_win))

            if z[1] > nz:
                continue

            sub_image = data[z[0]:z[1], y[0]:y[1], x[0]:x[1]] * apod
            chunk.append(sub_image)

        if HAS_CUPY and use_gpu:
            chunk_arr = cupy.array(chunk)
            apod_end = cupy.array(apod)
            betas = cupy.array(betas)
        else:
            chunk_arr = np.array(chunk)
            apod_end = apod.copy()

        fourier_image = fftn(chunk_arr, axes=(1, 2, 3))

        filt = get_filter(chunk_arr, fourier_image, betas,
                          half_win, **filter_kwargs)

        inverse_ft = (ifftn(fourier_image*filt,
                            axes=(1, 2, 3)
                            )*apod_end[np.newaxis, ]).real

        # fill all sub-images in z direction
        for ind, k in enumerate(strides_z):
            z = (int(k-half_win), int(k+half_win))

            if z[1] > nz:
                continue

            sub_image = cupy.asnumpy(inverse_ft[ind]) if (HAS_CUPY and use_gpu)\
                                                      else inverse_ft[ind]
            gated_data[z[0]:z[1], y[0]:y[1], x[0]:x[1]] += sub_image

        print('{:3.1f} % complete'.format(100*count/n_calc), end="\r")

    # correction factor for windowing
    # only applicable for sin^4 window
    gated_data /= (1.5)**3

    return gated_data


def noise_gate(data, perc=50, use_gpu=True, kwargs_filter={}, kwargs_noise={}):
    """
    Noise-gate a data cube.

    Parameters
    ----------
    data : ndarray
     Data cube - expects order (t, y, x)
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

    if not kwargs_filter:
        kwargs_filter = {'gamma': 1.5,
                         'gate': False}

    if not kwargs_noise:
        kwargs_noise = {'beta_const': False,
                        'image_indepen_noise': False,
                        'win_size': 12}

    betas = calc_noise_profile(data, perc=perc, **kwargs_noise)

    return run_filter(data, betas, use_gpu=use_gpu,
                      **kwargs_filter, **kwargs_noise)
