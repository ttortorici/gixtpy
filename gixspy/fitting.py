import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy import integrate


def gaussian0(x, amplitude, center, fwhm):
    """
    Standard gaussian function (normal distribution).
    :param x: x-axis array.
    :param amplitude: height of gaussian.
    :param center: center of gaussian.
    :param sigma: standard deviation of the mean.
    :return: y-axis array.
    """
    return amplitude * np.exp(-4 * np.log(2) * ((x - center) / fwhm) ** 2)


def gaussian(x, amplitude, center, fwhm, offset):
    """
    Standard gaussian function (normal distribution).
    :param x: x-axis array.
    :param amplitude: height of gaussian.
    :param center: center of gaussian.
    :param sigma: standard deviation of the mean.
    :return: y-axis array.
    """
    return amplitude * np.exp(-4 * np.log(2) * ((x - center) / fwhm) ** 2) + offset


def gaussian_double_peak(x, amplitude1, amplitude2, center1, center2, fwhm1, fwhm2, offset):
    """
    Standard gaussian function (normal distribution).
    :param x: x-axis array.
    :param amplitude: height of gaussian.
    :param center: center of gaussian.
    :param sigma: standard deviation of the mean.
    :return: y-axis array.
    """
    return amplitude1 * np.exp(-4 * np.log(2) * ((x - center2) / fwhm2) ** 2) + \
        amplitude2 * np.exp(-4 * np.log(2) * ((x - center2) / fwhm2) ** 2) + offset


def gaussian_triple_peak(x, amplitude1, amplitude2, amplitude3, center1, center2, center3,
                         fwhm1, fwhm2, fwhm3, offset):
    """
    Standard gaussian function (normal distribution).
    :param x: x-axis array.
    :param amplitude: height of gaussian.
    :param center: center of gaussian.
    :param sigma: standard deviation of the mean.
    :return: y-axis array.
    """
    return (amplitude1 * np.exp(-4 * np.log(2) * ((x - center1) / fwhm1) ** 2) +
            amplitude2 * np.exp(-4 * np.log(2) * ((x - center2) / fwhm2) ** 2) +
            amplitude3 * np.exp(-4 * np.log(2) * ((x - center3) / fwhm3) ** 2) + offset)


def gaussian_full_integral(amp, fwhm):
    return amp * fwhm * np.sqrt(2 * np.pi) / (2 * np.sqrt(2 * np.log(2)))


def gaussian_fit(x, y):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / sum(y))
    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    params, _ = curve_fit(f=gaussian, xdata=x, ydata=y, p0=[np.max(y), mean, fwhm, np.min(y)])
    return params


def gaussian_double_fit(x, y):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / sum(y))
    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    params, _ = curve_fit(f=gaussian_double_peak, xdata=x, ydata=y,
                          p0=[np.max(y), np.max(y), mean, mean, fwhm, fwhm, np.min(y)])
    return params


def gaussian_triple_fit(x, y):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / sum(y))
    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    params, _ = curve_fit(f=gaussian_triple_peak, xdata=x, ydata=y,
                          p0=[np.max(y), np.max(y), np.max(y), mean, mean, mean, fwhm, fwhm, fwhm, np.min(y)])
    return params


def gaussian_constrained_fit(x, y):
    """
    Fit data to a gaussian distribution with no offset and preserving area under the curve
    :param x: x-data array.
    :param y: y-data array.
    :return: (amplitude, center, sigma)
    """
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y))
    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    ls_res = least_squares(fun=residuals, x0=(np.max(y), mean, fwhm), args=(x, y))
    return ls_res.x


def residuals(p, x, y):
    """
    Residuals function to use for the least squares fit to apply the constraint of the total counts.
    :param p: Fit parameters (amplitude, center, sigma).
    :param x: x-data.
    :param y: y-data.
    :return: something to minimize.
    """
    def g_fit_func(argv):
        return gaussian0(argv, *p)

    integral = integrate.quad(g_fit_func, x[0], x[-1])[0]

    penalization = abs(np.sum(y) - integral) * 10000
    # return y - func1(x, p[0], p[1], p[2], p[3]) - penalization
    return y - gaussian0(x, *p) - penalization
