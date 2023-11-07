import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from loader import load_files


def main(files, pixel_size_x=.075, pixel_size_z=.075, cutoff=10):
    """

    :param files:
    :param pixel_size_x: in mm.
    :param pixel_size_z: in mm.
    :return:
    """
    # angles is a 1D array of length files-1
    # intensity_data is a 3D array of size (files-1, x-pixels, z-pixels)
    # intensity_db is a 2D array of size (x-pixels, z-pixels
    angles, intensity_data, intensity_db = load_files(files)
    x = np.arange(intensity_db.shape[1]) * pixel_size_x + (0.5 * pixel_size_x)
    z = np.arange(intensity_db.shape[0]) * pixel_size_z + (0.5 * pixel_size_z)
    z = z[::-1]
    x = x[425:525]
    z = z[700:850]
    # intensity_db = np.sum(intensity_db, axis=1)
    # intensity_data = np.sum(intensity_data, axis=1)
    intensity_db[np.where(intensity_db > cutoff)] = cutoff
    intensity_db = intensity_db[700:850, 425:525]

    intensity_data[np.where(intensity_data > cutoff)] = cutoff
    intensity_data = intensity_data[:, 700:850, 425:525]

    # plot_single_exposure(x, z, intensity_db)
    # plot_single_exposure(x, z, intensity_data[20])
    # plot_single_line(z, np.sum(intensity_data, axis=2)[20])
    plot_all(angles, z, np.sum(intensity_data, axis=2))


    plt.show()


def plot_single_line(z, intensity):
    plt.figure()
    plt.scatter(z, intensity)
    plt.title("Horizontal integration")
    plt.xlabel("z (mm)")
    plt.ylabel("Counts")


def plot_single_exposure(x, z, intensity):
    # intensity_data[np.where(intensity_data > cutoff)] = cutoff
    x, z = np.meshgrid(x, z)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    surf = ax.plot_surface(x, z, intensity, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_zlabel("Intensity")


def plot_all(angle, z, intensity):
    angle, z = np.meshgrid(angle, z)

    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    fig, ax = plt.subplots()
    ax.pcolor(angle, z, intensity.T, cmap=cm.coolwarm)

    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("Z (mm)")
    ax.set_label("Intensity")



if __name__ == '__main__':
    from loader import get_everthing, search_files
    tifs = get_everthing("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10")
    # tifs = get_everthing("D:\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10")
    main(tifs)
