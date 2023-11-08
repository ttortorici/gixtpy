import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation


def show():
    plt.show()


def tiff(tiff_data: np.ndarray, clip: int | float | None = None, log: bool = False) -> None:
    """
    Display a TIFF image.
    :param tiff_data: 2D numpy array containing pixel counts data (z, x)
    :param clip: an optional counts scale data clipping.
    :param log: optionally set log scale to counts.
    :return: None
    """
    if clip is not None:
        tiff_data[tiff_data > clip] = clip
    if log:
        tiff_data[tiff_data == 0] = 1
        tiff_data = np.log(tiff_data)
    plt.imshow(tiff_data)
    return None


def animate(intensity_data: np.ndarray, frame_delay: int = 20, clip: int | float | None = None, log: bool = False):
    """
    Animate TIFFs where each frame of the animation is a different exposure.
    :param intensity_data: 3D numpy array (exposure, z, x).
    :param frame_delay: time in milliseconds between each frame.
    :param clip: an optional counts scale data clipping.
    :param log: optionally set log scale to counts.
    :return: figure, animation (need these to successfully display)
    """
    if clip is not None:
        intensity_data[intensity_data > clip] = clip
    if log:
        intensity_data[intensity_data == 0] = 1
        tiff_data = np.log(intensity_data)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, intensity_data[0].shape[1]),
                  ylim=(0, intensity_data[0].shape[0]))
    im = plt.imshow(intensity_data[0][::-1], interpolation='none', vmin=0, vmax=10)

    def init():
        # im.set_data(intensity_db[z_lo:z_hi, x_lo:x_hi][::-1])
        im.set_data(intensity_data[0][::-1])
        return im,

    # animation function.  This is called sequentially
    def update(ii):
        im.set_array(intensity_data[ii][::-1])
        return im,

    ani = FuncAnimation(fig, update, frames=intensity_data.shape[0],
                        init_func=init, interval=frame_delay, blit=True)

    return fig, ani


def plot_line(intensity, pixel_size=None):
    plt.figure()
    intensity = np.sum(intensity, axis=1)
    z = np.arange(len(intensity))
    if pixel_size is not None:
        z = z.astype("f")
        z *= pixel_size
        plt.xlabel("z (mm)")
    else:
        plt.xlabel("z (pixels)")
    plt.scatter(z, intensity, facecolors="none", edgecolors="k", marker="o")
    plt.title("Horizontal integration")
    plt.ylabel("Counts")


def plot_line_fit(intensity, fit_func, fit_params, pixel_size=None):
    plt.figure()
    z = np.arange(len(intensity))
    if pixel_size is not None:
        z = z.astype("f")
        z *= pixel_size
        plt.xlabel("z (mm)")
    else:
        plt.xlabel("z (pixels)")
    plt.scatter(z, intensity, facecolors="none", edgecolors='k', marker="o")
    x = np.linspace(z[0], z[-1], 1000)
    plt.plot(x, fit_func(x, *fit_params))
    plt.title("Fit")

    plt.ylabel("Counts")


def plot_lines(z, intensities, angles):
    plt.figure()
    for angles, intensity in zip(angles, intensities):
        plt.scatter(z, intensity, label=f"{angles} degrees")
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


def plot_all(angle, intensity, pixel_size, clip, log):
    intensity = np.sum(intensity, axis=2)

    if clip is not None:
        intensity[intensity > clip] = clip
    if log:
        intensity[intensity == 0] = 1
        intensity = np.log(intensity)

    z_px = np.arange(intensity.shape[1])
    z = (z_px[::-1] + 0.5) * pixel_size
    angle, z = np.meshgrid(angle, z)

    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    fig, ax = plt.subplots()
    c = ax.pcolor(angle, z, intensity.T, cmap=cm.coolwarm)

    fig.colorbar(c, ax=ax)

    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("Z (mm)")
    ax.set_label("Intensity")
