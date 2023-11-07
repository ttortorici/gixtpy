import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy import integrate
import matplotlib.pyplot as plt
from loader import get_everthing, load_and_crop


def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x - center) ** 2 / (2. * sigma ** 2))


def gaussian_fit(x, y, c):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / sum(y))
    popt1, pcov1 = curve_fit(f=gaussian, xdata=x, ydata=y, p0=[max(y), mean, sigma])
    # popt1, pcov1 = curve_fit(func1, x, y)
    ls_res = least_squares(fun=residuals, x0=(max(y), mean, sigma), args=(x, y, c))
    print("curve_fit: {}".format(popt1))
    print("least_squares:")
    # print(ls_res)
    print(ls_res.x)
    print("...\n...")
    return popt1, ls_res.x


def residuals(p, x, y, total_counts):
    # sigma = p[-1]
    # amp = p[0]
    # integral = amp * sigma * np.sqrt(2 * np.pi)
    def g_fit_func(argv):
        return gaussian(argv, *p)

    integral = integrate.quad(g_fit_func, x[0], x[-1])[0]

    penalization = abs(total_counts - integral) * 10000
    # return y - func1(x, p[0], p[1], p[2], p[3]) - penalization
    return y - gaussian(x, *p) - penalization


def sech2(x, amplitude, center, sigma):
    return amplitude / np.cosh(np.pi * (x - center) / (2. * sigma)) ** 2


def sech2_fit(x, y):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(f=sech2, xdata=x, ydata=y, p0=[max(y), mean, sigma])
    print(popt)
    print(pcov)
    return popt


def test_guass_vs_sech_fit():
    tifs = get_everthing("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10")

    pixel_int_width = 5
    angles, intensity_data, intensity_db = load_and_crop(tifs, pixel_int_width)
    print("Total counts = {}".format(np.sum(intensity_db)))
    z = np.arange(intensity_db.shape[0])
    x = np.arange(intensity_db.shape[1])
    db = np.sum(intensity_db, axis=1)
    plt.plot(z, db)
    # print(np.array_repr(db2).replace("\n", "").replace("],", "],\n"))
    # print(db2[:, 13])
    gfit = gaussian_fit(z, db)

    g_sigma = gfit[-1]
    g_amp = gfit[0]
    integral_gauss = g_amp * g_sigma * np.sqrt(2 * np.pi)
    print("Gaussian countes = {}".format(integral_gauss))

    def g_fit_func(x):
        return gaussian(x, *gfit)

    print("Guassian integration = {}".format(integrate.quad(g_fit_func, 0, 1000)))

    sfit = sech2_fit(z, db)

    def s_fit_func(x):
        return sech2(x, *sfit)

    print("Sech2 integration = {}".format(integrate.quad(s_fit_func, 0, 1000)))


    # plt.plot(np.arange(len(db2[:, 13]))+825, db2[:, 13])
    x = np.linspace(0, z[-1], 100000)
    plt.plot(x, gaussian(x, *gfit), label="gauss")
    plt.plot(x, sech2(x, *sfit), label="sech2")
    plt.legend()


def test_guass_fit():
    tifs = get_everthing("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10")

    pixel_int_width = 1
    angles, intensity_data, intensity_db = load_and_crop(tifs, pixel_int_width)
    print("Total counts = {}".format(np.sum(intensity_db)))
    z = np.arange(intensity_db.shape[0])
    x = np.arange(intensity_db.shape[1])
    db = np.sum(intensity_db, axis=1)
    plt.plot(z, db)
    # print(np.array_repr(db2).replace("\n", "").replace("],", "],\n"))
    # print(db2[:, 13])
    gfit1, gfit2 = gaussian_fit(z, db, np.sum(intensity_db))


    g_sigma = gfit1[-1]
    g_amp = gfit1[0]
    integral_gauss = g_amp * g_sigma * np.sqrt(2 * np.pi)
    print("Gaussian1 counts = {}".format(integral_gauss))
    g_sigma = gfit2[-1]
    g_amp = gfit2[0]
    integral_gauss = g_amp * g_sigma * np.sqrt(2 * np.pi)
    print("Gaussian2 counts = {}".format(integral_gauss))

    x = np.linspace(0, z[-1], 100000)
    plt.plot(x, gaussian(x, *gfit1), label="gauss1")
    plt.plot(x, gaussian(x, *gfit2), label="gauss2")
    plt.legend()


if __name__ == "__main__":
    # tifs = get_everthing("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10")
    # db_tif = 'C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10\\direct_beam.tif'
    # db_tif = 'D:\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10\\direct_beam.tif'
    test_guass_fit()
    # angles, intensity_data, intensity_db = load_and_clip(tifs)
    # z = np.arange(intensity_db.shape[0])
    # x = np.arange(intensity_db.shape[1])


    # plt.imshow(intensity_data[10, 700:850, 425:525])
    plt.show()