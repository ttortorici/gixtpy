import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from loader import load_files, crop_data


def show(tiff_data, clip=None):
    if clip is not None:
        tiff_data[tiff_data > clip] = clip
    plt.imshow(tiff_data)


def animate(intensity_data, frame_delay=500, clip=None):
    """

    :param files:
    :param frame_delay (ms)
    :return:
    """
    if clip is not None:
        intensity_data[intensity_data > clip] = clip

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


if __name__ == '__main__':
    from loader import get_everthing, search_files

    tifs = get_everthing(
        "C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10")

    angles, intensity_data, intensity_db = load_files(tifs)
    intensity_data, intensity_db = crop_data(intensity_data, intensity_db, 30)

    # tifs = get_everthing("D:\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10")
    show(intensity_data[300], 300)
    # animate(intensity_data)
    plt.show()
