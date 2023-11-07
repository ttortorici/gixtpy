"""
To find critical angle for GIWAXS.

author: Teddy Tortorici
"""

import os
import matplotlib.pylab as plt
import numpy as np
from loader import (extract_angle_from, load_image, find_direct_beam_file_index)


# def show(fig):
#     plt.show()
#     fig.canvas.manager.window.activateWindow()
#     fig.canvas.manager.window.raise_()


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


def main(files: list[str] | tuple[str]) -> None:
    files = list(files)
    try:
        print(os.path.join(*files[0].split('/')[:-1]))
    except TypeError:
        print(os.path.join(*files[0].split('\\')[:-1]))

    direct_beam_file_index = find_direct_beam_file_index(files)
    print(f"Index of direct-beam file: {direct_beam_file_index}.")
    intensity_db = np.sum(load_image(files[direct_beam_file_index]), axis=1)
    del(files[direct_beam_file_index])
    direct_beam_indx = find_direct_beam_pixels_z(intensity_db)
    print(f"Pixel rows of direct beam: {direct_beam_indx}")

    """Plot direct beam intesnity (horizontal integration)"""
    plt.figure()
    plt.plot(intensity_db)
    plt.xlabel("Pixel row (counting from top)")
    plt.ylabel("Counts")
    plt.title("Direct Beam")

    """Plot reflected beams (horizontal integration)"""
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    angles = np.empty(len(files))
    total_refl_counts = np.empty(len(files))
    for ii, tif in enumerate(files):
        angle = str(extract_angle_from(tif))
        angles[ii] = float(angle)
        intensity = np.sum(load_image(tif), axis=1) - intensity_db
        intensity[intensity < 0] = 0.
        intensity[direct_beam_indx] = 0.
        total_refl_counts[ii] = np.sum(intensity)
        ax.plot(intensity, label=angle)
    # print(total_counts)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.title("Reflected beams (Horizontal Integration)")
    plt.xlim([direct_beam_indx[0] - 80, direct_beam_indx[-1]])
    plt.xlabel("Pixel y-axis (from top of array)")
    plt.ylabel("Counts")

    """Plot occluded beams (horizontal integration)"""
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    start_ind = direct_beam_indx[0] - 5
    end_ind = direct_beam_indx[-1] + 5

    plt.plot(intensity_db[start_ind:end_ind], label="direct beam")

    total_occl_counts = np.empty(len(files))
    for ii, tif in enumerate(files):
        angle = str(extract_angle_from(tif))
        angles[ii] = float(angle)
        intensity = np.sum(load_image(tif), axis=1)[start_ind:end_ind]
        total_occl_counts[ii] = np.sum(intensity)
        ax.plot(intensity, label=angle)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.title("Direct Beam occlusion (Horizontal Integration)")
    plt.xlabel("Pixel y-axis (from top of array)")
    plt.ylabel("Counts")

    """Plot total reflected intensity at each angle"""
    plt.figure()
    plt.scatter(angles, total_refl_counts)
    plt.title("Total Reflection Counts")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Counts")

    """Plot total occluded direct beam intensity at each angle"""
    plt.figure()
    plt.scatter(angles, total_occl_counts)
    plt.title("Total Occluded Direct Beam Counts")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Counts")
    return None


if __name__ == '__main__':
    from loader import get_everthing, search_files
    # tifs = search_files("D:\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\example_tune_data\\alkdjflas")
    # tifs = get_everthing("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\Capacitors\\Mounts\\Mount 01\\GIWAXS 2023-06\\2023-08-28\\om scan")
    # tifs = get_everthing("D:\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\Capacitors\\Mounts\\Mount 01\\GIWAXS 2023-08-28\\om scan")
    # tifs = get_everthing("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\Capacitors\\Mounts\\Mount 01\\GIWAXS 2023-09-27")
    # tifs = get_everthing("D:\\OneDrive - school\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\example_tune_data")
    tifs = get_everthing("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\XRD\\Silicon-silica-TT5-GIWAXS tune 2023-10-10")
    main(tifs)
    plt.show()
