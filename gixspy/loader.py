import os
import tkinter.filedialog
import numpy as np
import tifffile as tiff



def get_everthing(path: str) -> list[str]:
    """
    Get all files in a directory.
    :param path: Directory to get list of files.
    :return: List of all files in the directory.
    """
    filenames = os.listdir(path)
    return [os.path.join(path, name) for name in filenames]


def search_files(path: str = None) -> tuple[str]:
    """
    Open dialog to select files.
    :param path: Default path in dialog.
    :return: Tuple of files that were selected.
    """
    if path is None:
        path = os.getcwd()
    return tkinter.filedialog.askopenfilenames(title="Select TIF files", initialdir=path,
                                               filetypes=[("TIF", ".tif"), ("All files", "*")])


def load_and_crop(tifs: list | tuple, width: int) -> tuple:
    """
    Load files and crop the image around the direct beam.
    :param tifs: list of tifs
    :param width: number of pixels to include horizontally
    :return: angles, intensity_data, intensity_db
    """
    angles, intensity_data, intensity_db = load_files(tifs)
    intensity_data, intensity_db = crop_data(intensity_data, intensity_db, width=width)
    return angles, intensity_data, intensity_db


def load_crop_hori_sum(tifs: list | tuple, width: int) -> tuple:
    """
    Load files and crop the image around the direct beam.
    :param tifs: list of tifs
    :param width: number of pixels to include horizontally
    :return: angles, intensity_data, intensity_db
    """
    angles, intensity_data, intensity_db = load_and_crop(tifs, width)
    return angles, np.sum(intensity_data, axis=2), np.sum(intensity_db, axis=1)



def load_image(filename: str) -> np.ndarray:
    """
    Load a TIF file as data.
    :param filename: Full path to TIF file.
    :return: Pixel intensity in a 2D numpy array
    """
    image = np.array(tiff.imread(filename), dtype=int)
    image[image < 0] = 0
    return image


def extract_angle_from(filename: str) -> float:
    """
    Get the value of the angle from the name of a file.
    :param filename: The file in question.
    :return: The angle (in degrees) the data was taken at.
    """
    left_of_decimal = filename.split("_")[-3]
    angle = float(left_of_decimal)
    right_of_decimal = filename.split("_")[-2].replace(".tif", "")
    if left_of_decimal[0] == "-":
        angle -= float(right_of_decimal) / 10. ** len(right_of_decimal)
    else:
        angle += float(right_of_decimal) / 10. ** len(right_of_decimal)
    angle = round(angle, 3)
    return angle


def find_direct_beam_file_index(files: list[str] | tuple[str]) -> int:
    """
    Get the index of the file which contains the direct beam data.
    :param files: list of files.
    :return: index of the direct beam file.
    """
    for ii, tif in enumerate(files):
        if tif.split("/")[-1].replace(".tif", "").lower() in ("db", "direct_beam"):
            return ii
    for ii, tif in enumerate(files):
        if tif.split("\\")[-1].replace(".tif", "").lower() in ("db", "direct_beam"):
            return ii


def crop_data(exposure_datas, direct_beam_data, width, above=50, below=20):
    db_loc_x, db_loc_z = find_direct_beam_pixels(direct_beam_data)
    z_lo = db_loc_z[0] - above
    z_hi = db_loc_z[1] + below
    x_ave = (db_loc_x[0] + db_loc_x[1]) / 2

    x_lo = int(x_ave - width * 0.5)
    x_hi = round(x_ave + width * 0.5)
    # x_lo = db_loc_x[0] - 5
    # x_hi = db_loc_x[1] + 5
    exposure_datas = exposure_datas[:, z_lo:z_hi, x_lo:x_hi]
    direct_beam_data = direct_beam_data[z_lo:z_hi, x_lo:x_hi]
    # print(exposure_datas.shape)
    return exposure_datas, direct_beam_data


def find_direct_beam_pixels(direct_beam_tiff: np.ndarray, threshold: int = 8) -> tuple:
    """
    Get list of indices where the direct beam is.
    :param direct_beam_tiff: full tiff
    :param threshold: the count threshold to look for the beam
    :return: list of direct beam locations.
    """
    direct_beam_bool = direct_beam_tiff > threshold     # bool of pixel brightness above a threshold.
    direct_beam_indx_z, direct_beam_indx_x = np.where(direct_beam_bool)  # indices of those bright pixels in z direction
    # print(len(direct_beam_indx_x))
    # print(len(direct_beam_indx_z))

    distances_z = np.diff(direct_beam_indx_z)               # get distances of a bright pixel from the next
    distances_x = np.diff(direct_beam_indx_x)  # get distances of a bright pixel from the next
    # direct_beam_indx_x[get_indices_longest_repeated_element(distances_x)]
    # direct_beam_indx_z[get_indices_longest_repeated_element(distances_z)]
    longest_repeat_x = direct_beam_indx_x[get_indices_longest_repeated_element(distances_x)]
    longest_repeat_z = direct_beam_indx_z[get_indices_longest_repeated_element(distances_z)]

    x_start = longest_repeat_x[0]
    x_end = longest_repeat_x[-1]
    z_start = longest_repeat_z[0]
    z_end = longest_repeat_z[-1]
    # print((x_start + x_end)/2.)

    return (x_start, x_end), (z_start, z_end)


def find_direct_beam_pixels_z(direct_beam_data: np.ndarray, threshold: int = 8) -> np.ndarray:
    """
    Get list of indices where the direct beam is.
    :param direct_beam_data: horizontally integrated data set of direct beam exposure.
    :param threshold: the count threshold to look for the beam
    :return: list of direct beam locations.
    """
    direct_beam_bool = direct_beam_data > threshold     # bool of pixel brightness above a threshold.
    direct_beam_indx = np.where(direct_beam_bool)[0]    # indices of those bright pixels

    distances = np.diff(direct_beam_indx)               # get distances of a bright pixel from the next
    return direct_beam_indx[get_indices_longest_repeated_element(distances)]


def get_indices_longest_repeated_element(array: np.ndarray) -> np.ndarray:
    """
    Find the indices of the longest repeated element in an array.
    :param array: Array to sift through.
    :return: List of indices of locations of the longest repeated element
    """
    return find_sequence(array, get_longest_repeated_element(array))


def get_longest_repeated_element(arr: np.ndarray) -> np.ndarray:
    """
    Return the longest repeated sequence of an element in an array.
    :param arr: array to look through.
    :return: longest sequence of repeated elements
    """
    mask = np.concatenate(([False], np.diff(arr) == 0, [False]))
    idx = np.flatnonzero(mask[1:] != mask[:-1])
    islands = [arr[idx[i]:idx[i+1] + 1] for i in range(0, len(idx), 2)]
    island_lengths = np.array([len(island) for island in islands])
    return islands[np.argmax(island_lengths)]


def find_sequence(arr: np.ndarray, seq: np.ndarray) -> np.ndarray:
    """
    Search array for a specific sequence
    :param arr: array to search
    :param seq: array to find locations of
    :return: array of indices of where the sequence lies in the array
    """
    n_arr = arr.size
    n_seq = seq.size
    r_seq = np.arange(n_seq)
    match = (arr[np.arange(n_arr - n_seq + 1)[:, None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if match.any() > 0:
        return np.where(np.convolve(match, np.ones(n_seq, dtype=int)) > 0)[0]
    else:
        return np.array([])  # No match found


def load_files(files: list[str] | tuple[str]) -> tuple:
    """

    :param files:
    :param clipping_level:
    :return: angles, intensity_data, intensity_db
    """
    files = list(files)
    try:
        print(os.path.join(*files[0].split('/')[:-1]))
    except TypeError:
        print(os.path.join(*files[0].split('\\')[:-1]))

    direct_beam_file_index = find_direct_beam_file_index(files)
    # print(f"Index of direct-beam file: {direct_beam_file_index}.")
    intensity_db = load_image(files[direct_beam_file_index])
    intensity_db[np.where(intensity_db < 0)] = 0
    del(files[direct_beam_file_index])

    angles = np.empty(len(files))
    intensity_data = np.empty((len(files), *intensity_db.shape))
    for ii, tif in enumerate(files):
        angles[ii] = float(extract_angle_from(tif))
        intensity_data[ii] = load_image(tif)
        intensity_data[ii][np.where(intensity_data[ii] < 0)] = 0
        # intensity_data[ii][np.where(intensity_data[ii] > clipping_level)] = clipping_level


    # sort data by angles
    sorting_args = np.argsort(angles)
    angles = angles[sorting_args]
    intensity_data = intensity_data[sorting_args]

    return angles, intensity_data, intensity_db

