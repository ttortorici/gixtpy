import numpy as np
import fabio
import yaml
import pyFAI.detectors as detectors
from scipy.optimize import curve_fit
import pickle
from pathlib import Path
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class SpecularOmega:

    MAX_INT = (1 << 32) - 1

    def __init__(self, data_directory, det_dist=150, anglular_range=1.5, beam_width=1):
        self.z0 = 0
        self.det_dist=det_dist
        self.beam_width = beam_width
        self.data_directory = data_directory
        self.instrument = None
        self.detector = None
        self.file_type = None
        self.beam_center = None
        self.determine_instrument()

        self.file_list = list(data_directory.glob("*" + self.file_type))

        self.pixel_size = self.detector.get_pixel1() * 1e3  # mm/pixel
        self.pixels_above = int(det_dist * np.radians(anglular_range) / self.pixel_size)   # number of pixels above the beam to keep (for crop)
        self.pixels_below = 8   # number of pixels below the beam to keep (for crop)
        self.pixel_horiontal_offset = 0
        self.z = np.arange(self.detector.shape[0])[::-1] * self.pixel_size
        
        self.base_mask = np.logical_not(self.detector.calc_mask())

        self.run()

    def run(self):
        direct_beam_file_index = self.find_direct_beam_file_index()

        if direct_beam_file_index is not None:
            self.beam_center = self.find_beam_center(direct_beam_file_index)
            print("Direct beam is at ({}, {})".format(*self.beam_center))
            del(self.file_list[direct_beam_file_index])
        
        try:
            self.angles, self.intensity_data = self.load()
        except:
            self.angles,

        self.process_data()
        self.fit()
    
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
            elif fabio.open(file).header["Comment"].replace("Base - ", "").lower() in possible_identifiers:
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
        to_write = {"beamcenter": {'x': float(x0), 'y': float(y0)},
                    "sigma": {'x': float(sigma_x), 'y': float(sigma_y)},
                    "amplitude": {'x': float(a_x), 'y': float(a_y)}}
        with open(self.data_directory.parent / "beam_center.yaml", 'w') as yaml_file:
            yaml.dump(to_write, yaml_file, default_flow_style=False)
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
    
    def get_angle(self, filename: Path) -> float:
        if self.file_type == ".edf":
            angle = fabio.open(filename).header["Comment"].replace("Base - ", "")
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
    
    def process_data(self):
        if self.beam_center is not None:
            self.z -= (self.detector.shape[0] - self.beam_center[0]) * self.pixel_size
            bc_z = round(self.beam_center[0])
            z_lo = bc_z - self.pixels_above
            z_hi = bc_z + self.pixels_below

            self.z = self.z[z_lo:z_hi]

            if self.beam_width is None:
                data = self.intensity_data[:, z_lo:z_hi, :]
            else:
                half_width = 0.5 * self.beam_width / self.pixel_size
                x_lo = round(self.beam_center[1] - half_width)
                x_hi = round(self.beam_center[1] + half_width)
                x_lo += self.pixel_horiontal_offset
                x_hi += self.pixel_horiontal_offset
                data = self.intensity_data[:, z_lo:z_hi, x_lo:x_hi]
        else:
            data = self.intensity_data
        self.intensity_specular = np.sum(data, axis=2)
        # print(self.intensity_specular.max())

    def fit(self, z0=None, max_angle=1, pixel_cut=None, figsize=(4, 3)):
        if z0 is not None:
            self.z0 = z0
        print(f"z\u2080 = {self.z0}")
        max_ind = np.argmax(self.intensity_specular, axis=0)
        where_max_angle = self.angles[max_ind]

        # Remove data below the beam center
        valid = np.where(np.logical_and(
            self.z > self.z0,
            self.z < self.det_dist * np.radians(max_angle) + self.z0
            ))
        z = self.z[valid]
        where_max_angle = where_max_angle[valid]
        if pixel_cut is not None:
            z = z[:-pixel_cut]
            where_max_angle = where_max_angle[:-pixel_cut]
        
        (self.omega0, self.det_dist_fit), pcov = curve_fit(self.specular_fit, where_max_angle, z, p0=[0, self.det_dist])
        perr = np.sqrt(np.diag(pcov))
        print("Fit results:")
        print(f"    \u03C9\u2080 = ({self.omega0} \u00B1 {perr[0]})\u00B0")
        print(f"    d\u209B = ({self.det_dist_fit} \u00B1 {perr[1]}) mm")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.scatter(where_max_angle, z, s=10, marker='o',
                   edgecolors='k', lw=.75, facecolor='w')
        omega = np.linspace(where_max_angle[-1] - 0.02, where_max_angle[0] + 0.02, 100)
        ax.plot(omega, self.specular_fit(omega, self.omega0, self.det_dist_fit), "r")
        ax.tick_params(axis='both', which='both', direction='in', right=True, top=True)
        ax.set_ylabel("$z$ (mm)")
        ax.set_xlabel("$\\omega$ motor position $(\\degree)$")
        ax.grid(linestyle='dotted')
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.set_title("where max pixel occurs")
        # ax.legend(title="pixel")
        annotation_text = f"$\\omega_0 = ({self.omega0:.4f} \\pm {perr[0]:.4f})^\\circ$\n$d_{{sd}} = ({self.det_dist_fit:.1f} \\pm {perr[1]:.1f})$ mm"
        ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        fig.tight_layout()
        return fig, ax

    def specular_fit(self, omega, omega0, det_dist):
        return det_dist * np.tan(2. * np.radians(omega - omega0)) + self.z0
    
    def zero_angle(self, omega, omega0, det_dist):
        return det_dist * np.tan(np.radians(omega - omega0)) + self.z0
    
    def yoneda(self, omega, omega0, det_dist, critical_angle):
        return det_dist * np.tan(np.radians(omega - omega0 + critical_angle)) + self.z0
    
    def refraction_pos1(self, omega, omega0, det_dist, critical_angle):
        alpha = np.radians(omega - omega0)
        alpha_sq = alpha * alpha
        critical_angle = np.radians(critical_angle)
        crit_sq = critical_angle * critical_angle
        refraction_angle_sq = (alpha_sq - crit_sq) / (1 - 0.5 * crit_sq * alpha_sq)
        return self.z0 + det_dist * np.tan(alpha - np.sqrt(refraction_angle_sq))
    
    def refraction_pos2(self, omega, omega0, det_dist, critical_angle):
        alpha = np.radians(omega - omega0)
        alpha_sq = alpha * alpha
        critical_angle = np.radians(critical_angle)
        crit_sq = critical_angle * critical_angle
        refraction_angle_sq = (alpha_sq - crit_sq) / (1 - 0.5 * crit_sq * alpha_sq)
        return self.z0 + det_dist * np.tan(alpha + np.sqrt(refraction_angle_sq))
    
    def refraction_neg(self, omega, omega0, det_dist, critical_angle):
        alpha = np.radians(omega - omega0)
        critical_angle = np.radians(critical_angle)
        crit_sq = critical_angle * critical_angle
        refraction_angle = np.sqrt(alpha * alpha * (1 - 0.5 * crit_sq) + crit_sq)
        return self.z0 + det_dist * np.tan(alpha + refraction_angle)

    def plot(self, title="", critical_angle=None, horizon=False, det_dist=None, omega0=None):
        if det_dist is None:
            det_dist = self.det_dist_fit
        if omega0 is None:
            omega0 = self.omega0
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.set_facecolor('k')

        # z = np.arange(intensity_x_total.shape[1])[::-1] * self.pixel_size
        ax.set_ylabel("Z (mm)")

        color_map = ax.pcolormesh(self.angles, self.z, self.intensity_specular.T,
                                  norm=LogNorm(1, self.intensity_specular.max()), cmap="plasma",
                                  rasterized=True)
        omega2 = np.linspace(omega0, omega0 + .75, 1000)
        ax.plot(omega2, self.specular_fit(omega2, omega0, det_dist), "white", linewidth=1, alpha=0.5)
        if horizon:
            ax.plot(omega2, self.zero_angle(omega2, omega0, det_dist), "white", linewidth=1, alpha=0.5)
        if critical_angle is not None:
            if isinstance(critical_angle, float):
                critical_angle = [critical_angle]
            last = None
            for crit in critical_angle:
                if crit == last:
                    omega1 = np.linspace(omega0 + crit, omega0 + crit + .05, 100)
                    ax.plot(omega1, self.refraction_pos2(omega1, omega0, det_dist, crit), "white", linewidth=1, alpha=0.5)
                else:
                    ax.plot(omega2, self.yoneda(omega2, omega0, det_dist, crit), "white", linewidth=1, alpha=0.5)
                    omega1 = np.linspace(omega0 + crit, self.angles[-1], 1000)
                    ax.plot(omega1, self.refraction_pos1(omega1, omega0, det_dist, crit), "white", linewidth=1, alpha=0.5)
                    omega1 = np.linspace(self.angles[0], omega0, 1000)
                    ax.plot(omega1, self.refraction_neg(omega1, omega0, det_dist, crit), "white", linewidth=1, alpha=0.5)
                    
                    # spec_at = self.specular_fit(crit + self.omega0, self.omega0, self.det_dist_fit)
                    # ax.plot([crit + self.omega0, crit + self.omega0], [spec_at + 0.6, spec_at + 1], "white", linewidth=.5, alpha=0.5)
                last = crit
        color_bar = fig.colorbar(color_map, ax=ax)

        ax.set_xlabel("$\\omega\\ (\\degree)$")
        ax.set_label("Counts in row")
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax


class SpecularZ(SpecularOmega):
    def __init__(self, data_directory, det_dist=150):
        self.bc_sigma = None
        self.bc_amp = None
        super().__init__(data_directory, det_dist, 0.5)
        
    def run(self):
        with open(self.data_directory.parent / "beam_center.yaml", 'r') as yaml_file:
            beam_center_data = yaml.safe_load(yaml_file)
        self.beam_center = (beam_center_data["beamcenter"]['y'], beam_center_data["beamcenter"]['x'])
        
        sigma_x = beam_center_data["sigma"]['x']
        a_x = beam_center_data["amplitude"]['x']
        sigma_y = beam_center_data["sigma"]['y']
        a_y = beam_center_data["amplitude"]['y']
        print("Loaded beam center: ({}, {})".format(*self.beam_center))

if __name__ == "__main__":
    # om = Data(Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\TPP Films\BTB-TPP\2024 Film Growth\Film 2\XRD\non-grazing on blank for comparison\tune\raw_tiffs"))
    path = Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\TPP Films\BTB-TPP\2024 Film Growth\Film 4 (east campus dsc)\GIWAXS\film1\alignment")
    spec = SpecularOmega(path / "spec")
    spec.plot(critical_angle=0.24)
    # z = SpecularZ(path / "z-scan")
    plt.show()