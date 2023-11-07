import numpy as np
import matplotlib.pyplot as plt
from loader import get_everthing, load_crop_hori_sum
from fitting import gaussian_constrained_fit, gaussian0, gaussian_fit, gaussian # gaussian_double_fit, gaussian_double_peak
from plot_2d import plot_all
from scipy.signal import argrelextrema


def find_reflection_index_start(angles, intensity_data):
    total_counts_each_angle = np.sum(intensity_data, axis=1)
    fit_p = gaussian_fit(angles, total_counts_each_angle)
    fit_min_ind = argrelextrema(total_counts_each_angle, np.less)
    # fit_min_ind = argrelextrema(gaussian_double_peak(angles, *fit_p), np.less)[0][0]
    # fit_min_ind = np.where(gaussian_offset(angles, *fit_p) < fit_p[-1] + 1)
    print(fit_min_ind)
    # print(angles[fit_min_ind])
    plt.figure()
    plt.plot(angles, total_counts_each_angle)
    plt.plot(angles, gaussian(angles, *fit_p))
    # return fit_min_ind


def find_reflection(angles, z_px, z, intensity_data, intensity_db):
    db_fit_params = gaussian_constrained_fit(z_px, intensity_db)  # amplitude, center, sigma
    db_first_pixel = np.where(gaussian0(z_px, *db_fit_params) > 1)[0][0]
    # print(db_first_pixel)

    intensity_data[:, db_first_pixel:] = 0.

    plot_all(angles, z, intensity_data)

    fit_min_ind = find_reflection_index_start(angles, intensity_data)

    plot_all(angles[fit_min_ind:], z, intensity_data[fit_min_ind:, :])


def main(files, tiff_clip=100, pixel_size_z=.075):
    """

    :param files:
    :param pixel_size_z: in mm.
    :return:
    """
    # angles is a 1D array of length files-1
    # intensity_data is a 2D array of size (files-1, z-pixel intensities)
    # intensity_db is a 1D array of size (z-pixel intensities)
    angles, intensity_data, intensity_db = load_crop_hori_sum(files, 25)
    print(f"Angles: {len(angles)}")
    print(f"Data set shape: {intensity_data.shape}")
    print(f"Number of DB pixels: {len(intensity_db)}")
    intensity_data[intensity_data > tiff_clip] = tiff_clip
    z_px = np.arange(len(intensity_db))
    z = (z_px[::-1] + 0.5) * pixel_size_z



    # Convert gaussian to mm
    # db_fit_params[1] += 0.5             # shift center to pixel center
    # db_fit_params[1:] *= pixel_size_z   # scale center and sigma to mm

    # intensity_data[:, db_first_pixel:] = 0.

    plot_all(angles, z, intensity_data)

    find_reflection(angles, z_px, z, intensity_data, intensity_db)

    # z_fit = np.linspace(z[0], z[-1], 1000)
    # plt.plot(z_fit[::-1], gaussian(z_fit, *db_fit_params))
    # plt.plot(z, intensity_db)
    plt.show()




if __name__ == '__main__':
    from loader import get_everthing, search_files

    tifs = get_everthing("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10")
    # tifs = get_everthing("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\example_tune_data")

    main(tifs, 100000)
