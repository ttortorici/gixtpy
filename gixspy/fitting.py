import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy import integrate
import gixspy.plot as plt


def direct_beam_line(direct_beam, plot=True):
    if len(direct_beam.shape) == 2:
        direct_beam = np.sum(direct_beam, axis=1)
    z_px = np.arange(len(direct_beam))
    fit_params = gaussian_constrained_fit(z_px, direct_beam)
    print(f"Amplitude = {fit_params[0]} counts")
    print(f"Center = {fit_params[1]} pixels")
    print(f"Full width half max = {fit_params[2]} pixels")
    print(f"Total counts in the data = {np.sum(direct_beam)} counts")
    print(f"Total counts in the fit = {gaussian_full_integral(fit_params[0], fit_params[2])} counts")
    # db_first_pixel = np.where(gaussian0(z, *gfit) > 1)[0][0]

    if plot:
        plt.plot_line_fit(direct_beam, gaussian0, fit_params)
    return fit_params


def reflection_line(direct_beam, plot=True):
    if len(direct_beam.shape) == 2:
        direct_beam = np.sum(direct_beam, axis=1)
    z_px = np.arange(len(direct_beam))
    fit_params = gaussian_fit0(z_px, direct_beam)
    print(f"Amplitude = {fit_params[0]} counts")
    print(f"Center = {fit_params[1]} pixels")
    print(f"Full width half max = {fit_params[2]} pixels")
    print(f"Total counts in the data = {np.sum(direct_beam)} counts")
    print(f"Total counts in the fit = {gaussian_full_integral(fit_params[0], fit_params[2])} counts")
    # db_first_pixel = np.where(gaussian0(z, *gfit) > 1)[0][0]

    if plot:
        plt.plot_line_fit(direct_beam, gaussian0, fit_params)
    return fit_params


def line(intensity, plot=True):
    if len(intensity.shape) == 2:
        intensity = np.sum(intensity, axis=1)
    z_px = np.arange(len(intensity))
    fit_params = gaussian_4_fit(z_px, intensity)
    for ii in range(1, 5):
        print(f"Amplitude{ii} = {fit_params[ii]} counts")
        print(f"Center{ii} = {fit_params[ii+1]} pixels")
        print(f"Full{ii} width half max = {fit_params[ii+2]} pixels")

    if plot:
        plt.plot_line_fit(intensity, gaussian0, fit_params)
    return fit_params


def gaussian0(x, amplitude, center, fwhm):
    """
    Standard gaussian function (normal distribution) with no offset.
    :param x: x-axis array.
    :param amplitude: height of gaussian.
    :param center: center of gaussian.
    :param fwhm: full width half max of gaussian.
    :return: y-axis array.
    """
    return amplitude * np.exp(-4 * np.log(2) * ((x - center) / fwhm) ** 2)


def gaussian(x, amplitude, center, fwhm, offset):
    """
    Standard gaussian function (normal distribution).
    :param x: x-axis array.
    :param amplitude: height of gaussian.
    :param center: center of gaussian.
    :param fwhm: full width half max of gaussian.
    :param offset: y-axis offset of gaussian.
    :return: y-axis array.
    """
    return amplitude * np.exp(-4 * np.log(2) * ((x - center) / fwhm) ** 2) + offset


def gaussian_2_peak(x, amplitude1, center1, fwhm1, amplitude2, center2, fwhm2, offset):
    """
    Two peaked gaussian function.
    :param x: x-axis array.
    :param amplitude1: height of 1st peak.
    :param center1: center of 1st peak.
    :param fwhm1: full width half max of 1st peak.
    :param amplitude2: height of 2nd peak.
    :param center2: center of 2nd peak.
    :param fwhm2: full width half max of 2nd peak.
    :param offset: y-axis offset of 2nd peak.
    :return: y-axis array.
    """
    return amplitude1 * np.exp(-4 * np.log(2) * ((x - center1) / fwhm1) ** 2) + \
        amplitude2 * np.exp(-4 * np.log(2) * ((x - center2) / fwhm2) ** 2) + offset


def gaussian_3_peak(x, amplitude1, center1, fwhm1, amplitude2, center2, fwhm2, amplitude3, center3, fwhm3, offset):
    """
    Three peaked gaussian function.
    :param x: x-axis array.
    :param amplitude1: height of 1st peak.
    :param center1: center of 1st peak.
    :param fwhm1: full width half max of 1st peak.
    :param amplitude2: height of 2nd peak.
    :param center2: center of 2nd peak.
    :param fwhm2: full width half max of 2nd peak.
    :param amplitude3: height of 3rd peak.
    :param center3: center of 3rd peak.
    :param fwhm3: full width half max of 3rd peak.
    :param offset: y-axis offset of 2nd peak.
    :return: y-axis array.
    """
    return (amplitude1 * np.exp(-4 * np.log(2) * ((x - center1) / fwhm1) ** 2) +
            amplitude2 * np.exp(-4 * np.log(2) * ((x - center2) / fwhm2) ** 2) +
            amplitude3 * np.exp(-4 * np.log(2) * ((x - center3) / fwhm3) ** 2) + offset)


def gaussian_4_peak(x, amplitude1, center1, fwhm1, amplitude2, center2, fwhm2,
                    amplitude3, center3, fwhm3, amplitude4, center4, fwhm4, offset):
    """
    Four peaked gaussian function.
    :param x: x-axis array.
    :param amplitude1: height of 1st peak.
    :param center1: center of 1st peak.
    :param fwhm1: full width half max of 1st peak.
    :param amplitude2: height of 2nd peak.
    :param center2: center of 2nd peak.
    :param fwhm2: full width half max of 2nd peak.
    :param amplitude3: height of 3rd peak.
    :param center3: center of 3rd peak.
    :param fwhm3: full width half max of 3rd peak.
    :param amplitude4: height of 3rd peak.
    :param center4: center of 3rd peak.
    :param fwhm4: full width half max of 3rd peak.
    :param offset: y-axis offset of 2nd peak.
    :return: y-axis array.
    """
    return (amplitude1 * np.exp(-4 * np.log(2) * ((x - center1) / fwhm1) ** 2) +
            amplitude2 * np.exp(-4 * np.log(2) * ((x - center2) / fwhm2) ** 2) +
            amplitude3 * np.exp(-4 * np.log(2) * ((x - center3) / fwhm3) ** 2) +
            amplitude4 * np.exp(-4 * np.log(2) * ((x - center4) / fwhm4) ** 2) + offset)


def gaussian_full_integral(amp, fwhm):
    return amp * fwhm * np.sqrt(2 * np.pi) / (2 * np.sqrt(2 * np.log(2)))


def gaussian_fit0(x, y):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / sum(y))
    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    params, _ = curve_fit(f=gaussian0, xdata=x, ydata=y, p0=[np.max(y), mean, fwhm],
                          bounds=((0, -np.inf, 0), (np.inf, np.inf, np.inf)))
    return params


def gaussian_fit(x, y):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / sum(y))
    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    params, _ = curve_fit(f=gaussian, xdata=x, ydata=y, p0=[np.max(y), mean, fwhm, np.min(y)],
                          bounds=((0, -np.inf, 0, 0), (np.inf, np.inf, np.inf, np.inf)))
    return params


def gaussian_2_fit(x, y):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / sum(y))
    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    params, _ = curve_fit(f=gaussian_2_peak, xdata=x, ydata=y,
                          p0=[np.max(y), mean, fwhm, np.max(y), mean, fwhm, np.min(y)],
                          bounds=((0, -np.inf, 0, 0, -np.inf, 0, 0),
                                  (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)))
    return params


def gaussian_3_fit(x, y):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / sum(y))
    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    params, _ = curve_fit(f=gaussian_3_peak, xdata=x, ydata=y,
                          p0=[np.max(y), mean, fwhm, np.max(y), mean, fwhm, np.max(y), mean, fwhm, np.min(y)],
                          bounds=((0, -np.inf, 0, 0, -np.inf, 0, 0, -np.inf, 0, 0),
                                  (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)))
    return params


def gaussian_4_fit(x, y):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / sum(y))
    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    params, _ = curve_fit(f=gaussian_4_peak, xdata=x, ydata=y,
                          p0=[np.max(y), mean, fwhm, np.max(y), mean, fwhm, np.max(y), mean, fwhm,
                              np.max(y), mean, fwhm, np.min(y)],
                          bounds=((0, -np.inf, 0, 0, -np.inf, 0, 0, -np.inf, 0, 0, -np.inf, 0, 0),
                                  (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                   np.inf, np.inf, np.inf)))
    return params


def gaussian_constrained_fit(x, y):
    """
    Fit data to a gaussian distribution with no offset and preserving area under the curve
    :param x: x-data array.
    :param y: y-data array.
    :return: (amplitude, center, fwhm)
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
