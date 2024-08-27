import numpy as np
import fabio
import pyFAI.detectors as detectors
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class Data:

    MAX_INT = (1 << 32) - 1

    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.instrument = None
        self.detector = None
        self.file_type = None
        self.determine_instrument()

        self.file_list = list(data_directory.glob("*" + self.file_type))

        self.pixels_above = 75   # number of pixels above the beam to keep (for crop)
        self.pixels_below = 10   # number of pixels below the beam to keep (for crop)
        self.pixel_horiontal_offset = 0
        self.pixel_size = self.detector.get_pixel1() * 1e3  # mm/pixel
        self.base_mask = np.logical_not(self.detector.calc_mask())

        direct_beam_file_index = self.find_direct_beam_file_index()

        self.beam_center = None
        if direct_beam_file_index is not None:
            self.beam_center = self.find_beam_center(direct_beam_file_index)
            print("Direct beam is at ({}, {})".format(*self.beam_center))
            del(self.file_list[direct_beam_file_index])

        self.angles, self.intensity_data = self.load()
    
    def determine_instrument(self):
        num_edf_files = len(list(self.data_directory.glob("*.edf")))
        num_tif_files = len(list(self.data_directory.glob("*.tif")))

        if num_edf_files > num_tif_files:
            self.instrument = "east"
            self.detector = detectors.Eiger2_1M()
            self.file_type = ".edf"
        else:
            self.instrument = "main"
            self.detector = detectors.Eiger1M()
            self.file_type = ".tif"

    def find_direct_beam_file_index(self):
        possible_identifiers = ("db", "direct_beam", "om_scan_direct_beam")
        for ii, file in enumerate(self.file_list):
            if self.file_type == ".tif":
                if file.name.replace(".tif", "").lower() in possible_identifiers:
                    return ii
            elif fabio.open(file).header["Comment"] in possible_identifiers:
                return ii
        print("Did not find a direct beam file")
        return None
    
    def find_beam_center(self, direct_beam_index_file: int) -> tuple:
        intensity_db = self.load_image(self.file_list[direct_beam_index_file])
        rows = self.detector.shape[0]
        columns = self.detector.shape[1]
        
        px_x = np.arange(columns)
        db_x = np.sum(intensity_db, axis=0)
        px_y = np.arange(rows)
        db_y = np.sum(intensity_db, axis=1)
        def gaussian(x, x0, sigma, amplitude):
            arg = (x - x0) / sigma
            return amplitude * np.exp(-0.5 * arg * arg)
        (x0, sigma_x, a_x), _, = curve_fit(gaussian, px_x, db_x,
                                           p0=(0.5 * columns, 100, db_x.max()),
                                           nan_policy="omit")
        (y0, sigma_y, a_y), _ = curve_fit(gaussian, px_y, db_y,
                                           p0=(0.75 * rows, 100, db_y.max()),
                                           nan_policy="omit")
        beam_center = (y0, x0)
        # beam_center = np.unravel_index(np.argmax(intensity_db), intensity_db.shape)
        fig, ax = plt.subplots(1, 1)
        ax.set_facecolor('k')
        pos = ax.imshow(intensity_db, norm=LogNorm(1, intensity_db.max()))
        ax.scatter(beam_center[1], beam_center[0], color='r')
        fig.colorbar(pos, ax=ax)
        return beam_center
    
    def load_image(self, filename: Path) -> np.ndarray:
        file_data = fabio.open(filename).data
        mask = self.base_mask.copy()
        if file_data.dtype == np.uint32:
            mask[np.where(file_data == self.MAX_INT)] = 0
        file_data *= mask
        return file_data
    
    def load(self):
        angles = np.empty(len(self.file_list))
        intensity_data = np.empty((len(self.file_list), *self.detector.shape))
        for ii, file in enumerate(self.file_list):
            angles[ii] = self.get_angle(file)
            intensity_data[ii] = self.load_image(file)

        sorting_args = np.argsort(angles)
        angles = angles[sorting_args]
        intensity_data = intensity_data[sorting_args]

        return angles, intensity_data

    def remove_crop(self):
        pixels_above = 75   # number of pixels above the beam to keep (for crop)
        pixels_below = 10   # number of pixels below the beam to keep (for crop)
        pixel_horiontal_offset = 0

    def crop(self, x_pixel_width, pixels_below, pixels_above, pixel_horiontal_offset=0):
        self.x_pixel_width = x_pixel_width
        self.pixels_below = pixels_below
        self.pixels_above = pixels_above
        self.pixel_horiontal_offset = pixel_horiontal_offset
        #  db_loc_z = round(self.beam_center[0])
        #  db_loc_x = round(self.beam_center[1])
        #  z_lo = db_loc_z[0] - pixels_above
        #  z_hi = db_loc_z[1] + pixels_below
        #  x_ave = (db_loc_x[0] + db_loc_x[1]) / 2
        #  
        #  x_lo = int(x_ave - x_pixel_width * 0.5)
        #  x_hi = round(x_ave + x_pixel_width * 0.5)
        #  x_lo += x_offset
        #  x_hi += x_offset
        #  exposure_datas = self.intensity_data[:, z_lo:z_hi, x_lo:x_hi]
        #  return exposure_datas
    
    def get_angle(self, filename: Path) -> float:
        if self.file_type == ".edf":
            angle = fabio.open(filename).header["Comment"]
        elif self.file_type == ".tif":
            angle = self.angle_from_filename(filename.name)
        return float(angle)

    @staticmethod
    def angle_from_filename(filename: str):
        left_of_decimal = filename.split("_")[-3]
        angle = float(left_of_decimal)
        right_of_decimal = filename.split("_")[-2].replace(".tif", "")
        if left_of_decimal[0] == "-":
            angle -= float(right_of_decimal) / 10. ** len(right_of_decimal)
        else:
            angle += float(right_of_decimal) / 10. ** len(right_of_decimal)
        angle = round(angle, 3)
        return angle
    
    def plot(self, beam_width=None, title=""):
        if self.beam_center is not None:
            bc_z = round(self.beam_center[0])
            z_lo = bc_z - self.pixels_above
            z_hi = bc_z + self.pixels_below

            if beam_width is None:
                data = self.intensity_data[:, z_lo:z_hi, :]
            else:
                half_width = 0.5 * beam_width / self.pixel_size
                x_lo = round(self.beam_center[1] - half_width)
                x_hi = round(self.beam_center[1] + half_width)
                x_lo += self.pixel_horiontal_offset
                x_hi += self.pixel_horiontal_offset
                data = self.intensity_data[:, z_lo:z_hi, x_lo:x_hi]
        else:
            data = self.intensity_data
        intensity_x_total = np.sum(data, axis=2)
        print(intensity_x_total.max())

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.set_facecolor('k')

        z = (np.arange(intensity_x_total.shape[1])[::-1]) * self.pixel_size
        ax.set_ylabel("Z (mm)")

        color_map = ax.pcolormesh(self.angles, z, intensity_x_total.T,
                                  norm=LogNorm(1, intensity_x_total.max()), cmap="plasma")

        color_bar = fig.colorbar(color_map, ax=ax)

        ax.set_xlabel("$\\omega\\ (\\degree)$")
        ax.set_label("Counts in row")
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax


if __name__ == "__main__":
    data = Data(Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\TPP Films\BTB-TPP\2024 Film Growth\Film 2\XRD\non-grazing on blank for comparison\tune\raw_tiffs"))
    plt.show()