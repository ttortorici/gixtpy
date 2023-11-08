import os
import tkinter.filedialog
import numpy as np
import tifffile as tiff


def tiff_filenames(path: str) -> list[str]:
    """
    Return list of all TIFF files in a specified directory.
    :param path: Directory to get list of files.
    :return: List of all TIFF files in the directory.
    """
    filenames = os.listdir(path)
    return [os.path.join(path, name) for name in filenames if ".tif" in name]


def search_tiffs(path: str = None) -> tuple[str]:
    """
    Open dialog to select files to form a list of those files.
    :param path: Path that dialog will default to (will be the working directory if None given).
    :return: Tuple of files that were selected.
    """
    if path is None:
        path = os.getcwd()
    return tkinter.filedialog.askopenfilenames(title="Select TIFF files", initialdir=path,
                                               filetypes=[("TIFF", ".tif"), ("All files", "*")])


def tiff_data(files: list[str] | tuple[str]) -> tuple:
    """
    Load all relevant data from a list of TIFFS
    :param files:
    :return: angles, intensity_data, intensity_db
    """
    files = list(files)
    try:
        print(os.path.join(*files[0].split('/')[:-1]))
    except TypeError:
        print(os.path.join(*files[0].split('\\')[:-1]))

    direct_beam_file_index = find_direct_beam_file_index(files)
    # print(f"Index of direct-beam file: {direct_beam_file_index}.")
    intensity_db = __load_image(files[direct_beam_file_index])
    intensity_db[np.where(intensity_db < 0)] = 0
    del(files[direct_beam_file_index])

    angles = np.empty(len(files))
    intensity_data = np.empty((len(files), *intensity_db.shape))
    for ii, tif in enumerate(files):
        angles[ii] = float(extract_angle_from(tif))
        intensity_data[ii] = __load_image(tif)
        intensity_data[ii][np.where(intensity_data[ii] < 0)] = 0

    sorting_args = np.argsort(angles)
    angles = angles[sorting_args]
    intensity_data = intensity_data[sorting_args]

    return angles, intensity_data, intensity_db


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


def crop_data(exposure_datas: np.ndarray, direct_beam_data: np.ndarray, width: int, above: int = 50, below: int = 20,
              x_offset: int = 0) -> tuple[np.ndarray]:
    """
    Crop TIFF data around where the direct beam appears.
    :param exposure_datas: 3D numpy array of counts of all the exposures (exposure, z, x).
    :param direct_beam_data: 2D numpy array of counts of the direct beam exposure (z, x).
    :param width: Number of horizontal pixels to keep centered around the direct beam center. Can estimate with
                  (beam horizontal shutter size [s2hg]) / (pixel width): e.g. (0.8 mm) / (0.075 mm/px) ~ 11 px.
    :param above: The number of pixels above the direct beam to keep (~50 recommended).
    :param below: The number of pixels below the direct beam to keep.
    :param x_offset: optional offset for the width center if you find that the beam center isn't being properly found.
    :return: cropped exposure data (3D array), cropped direct beam data (2D array).
    """

    db_loc_x, db_loc_z = __find_direct_beam_pixels(direct_beam_data)
    z_lo = db_loc_z[0] - above
    z_hi = db_loc_z[1] + below
    x_ave = (db_loc_x[0] + db_loc_x[1]) / 2

    x_lo = int(x_ave - width * 0.5)
    x_hi = round(x_ave + width * 0.5)
    x_lo += x_offset
    x_hi += x_offset
    exposure_datas = exposure_datas[:, z_lo:z_hi, x_lo:x_hi]
    direct_beam_data = direct_beam_data[z_lo:z_hi, x_lo:x_hi]
    # print(exposure_datas.shape)
    return exposure_datas, direct_beam_data


def __find_direct_beam_pixels(direct_beam_tiff: np.ndarray, threshold: int = 8) -> tuple[tuple[int]]:
    """
    Get list of indices where the direct beam is.
    :param direct_beam_tiff: full tiff
    :param threshold: the count threshold to look for the beam
    :return: list of direct beam locations.
    """
    direct_beam_bool = direct_beam_tiff > threshold     # bool of pixel brightness above a threshold.
    direct_beam_indx_z, direct_beam_indx_x = np.where(direct_beam_bool)  # indices of those bright pixels in z direction

    distances_z = np.diff(direct_beam_indx_z)               # get distances of a bright pixel from the next
    distances_x = np.diff(direct_beam_indx_x)  # get distances of a bright pixel from the next
    longest_repeat_x = direct_beam_indx_x[__get_indices_longest_repeated_element(distances_x)]
    longest_repeat_z = direct_beam_indx_z[__get_indices_longest_repeated_element(distances_z)]

    x_start = longest_repeat_x[0]
    x_end = longest_repeat_x[-1]
    z_start = longest_repeat_z[0]
    z_end = longest_repeat_z[-1]

    return (x_start, x_end), (z_start, z_end)


def __get_indices_longest_repeated_element(array: np.ndarray) -> np.ndarray:
    """
    Find the indices of the longest repeated element in an array.
    :param array: Array to sift through.
    :return: List of indices of locations of the longest repeated element
    """
    return __find_sequence(array, __get_longest_repeated_element(array))


def __get_longest_repeated_element(arr: np.ndarray) -> np.ndarray:
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


def __find_sequence(arr: np.ndarray, seq: np.ndarray) -> np.ndarray:
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


def __load_image(filename: str) -> np.ndarray:
    """
    Load a TIF file as data.
    :param filename: Full path to TIF file.
    :return: Pixel intensity in a 2D numpy array
    """
    image = np.array(tiff.imread(filename), dtype=int)
    image[image < 0] = 0
    return image