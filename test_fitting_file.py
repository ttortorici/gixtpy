import numpy as np
import matplotlib.pyplot as plt
from loader import get_everthing, load_and_crop, load_crop_hori_sum
from fitting import gaussian_constrained_fit, gaussian0, gaussian_inv


def test_guass_fit(width_pixels):
    tifs = get_everthing("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10")
    pixel_size = 0.075
    angles, intensity_data, db = load_crop_hori_sum(tifs, width_pixels)
    ind1 = np.where(angles == 0.62)[0][0]
    print(ind1)
    print("Total counts = {}".format(np.sum(db)))
    z = np.arange(len(db))
    # db = np.sum(intensity_db, axis=1)
    plt.plot(z, db, label="direct beam")

    gfit = gaussian_constrained_fit(z, db)
    db_first_pixel = np.where(gaussian0(z, *gfit) > 1)[0][0]

    for ii in range(8):
        index = ind1 + ii - 5

        intensity = intensity_data[index]
        intensity[db_first_pixel:] = 0
        plt.plot(z, intensity, label=angles[index])
    # print(np.array_repr(db2).replace("\n", "").replace("],", "],\n"))
    # print(db2[:, 13])


    g_sigma = gfit[-1]
    print("sigma {} mm".format(2. * np.sqrt(2. * np.log(2.)) * g_sigma * pixel_size))
    g_amp = gfit[0]
    integral_gauss = g_amp * g_sigma * np.sqrt(2 * np.pi)
    print("Gaussian1 counts = {}".format(integral_gauss))

    x = np.linspace(0, z[-1], 100000)
    plt.plot(x, gaussian0(x, *gfit), label="direct beam fit")
    plt.legend()
    plt.title("Refracted Beam")
    plt.xlabel("z-pixel (counting from top of CCD)")
    plt.ylabel("Counts")


if __name__ == "__main__":
    test_guass_fit(5)
    plt.show()
