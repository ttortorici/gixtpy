# gixtpy
## Grazing Incidence X-ray Tunning Python Library (pronounced: jist-pie)

Tools for assisting in tuning angle of incidence for grazing incidence x-ray scattering experiments.

### Example
```
import gixtpy

tiff_list = gixtpy.search_tiff_files()

angles, intensity_data, direct_beam = gixtpy.load_tiff_data(tiff_list)

print("angles shape: {}".format(angles.shape))
print("intensity_data shape: {}".format(intensity_data.shape))
print("direct_beam shape: {}".format(direct_beam.shape))

x_pixel_width = 15  # number of pixels to sum across horizontally centered around the beam center
pixels_above = 50   # number of pixels above the beam to keep (for crop)
pixels_below = 20   # number of pixels below the beam to keep (for crop)
pixel_size = 0.075  # mm/pixel

"""Crop data"""
id_c, db_c = gixtpy.crop_data(intensity_data, direct_beam, x_pixel_width, pixels_above, pixels_below)

"""Animate"""
fps = 24     # frames per second
clip = None  # counts clipping level
log = True   # animate on a log scale

fig, ani = gixtpy.animate_tiffs(id_c, fps, clip, log)  # must return figure and animation for it to display

"""Plot counts vs z vs counts"""
clip = None  # counts clipping level
log = True  # animate on a log scale
gixtpy.plot_tuning(angles, id_c, pixel_size, clip, log)
gixtpy.show()
```

### Functions

```
get_all_tiff_file_names(path: str) -> list[str]:
    """
    Return list of all TIFF files in a specified directory.
    :param path: Directory to get list of files.
    :return: List of all TIFF files in the directory.
    """
```

```
search_tiff_files(path: str = None) -> tuple[str]:
    """
    Open dialog to select files to form a list of those files.
    :param path: Path that dialog will default to (will be the working directory if None given).
    :return: Tuple of files that were selected.
    """
```

```
load_tiff_data(files: list[str] | tuple[str]) -> tuple:
    """
    Load all relevant data from a list of TIFFS
    :param files:
    :return: angles, intensity_data, intensity_db
    """
```

```
crop_data(exposure_datas: np.ndarray, direct_beam_data: np.ndarray, width: int, above: int = 50, below: int = 20,
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
```

```
show() -> None:
    """
    Show all plots.
    :return: None
    """
```

```
display_tiff(tiff_data: np.ndarray, clip: int | float | None = None, log: bool = False) -> None:
    """
    Display a TIFF image.
    :param tiff_data: 2D numpy array containing pixel counts data (z, x)
    :param clip: an optional counts scale data clipping.
    :param log: optionally set log scale to counts.
    :return: None
    """
```

```
animate_tiffs(intensity: np.ndarray, fps: float = 24.0, clip: int | float | None = None, log: bool = False,
              color_scheme: int = 0):
    """
    Animate TIFFs where each frame of the animation is a different exposure.
    :param intensity: 3D numpy array (exposure, z, x).
    :param fps: frames per second.
    :param clip: an optional counts scale data clipping.
    :param log: True will result in a log scale display of intensity.
    :param color_scheme: choice of color scheme (given with an int value).
    :return: figure, animation (need to return these to successfully display)
    """
```

```
plot_tuning(angle: np.ndarray, intensity: np.ndarray, pixel_size: float | None = None,
            clip: float | int | None = None, log: bool = False, color_scheme: int = 0) -> tuple:
    """
    Plot angle tuning data.
    :param angle: motor angle at each exposure.
    :param intensity: intensity data (exposure, z, x).
    :param pixel_size: optionally scale data to mm with pixel size in mm.
    :param clip: optionally clip intensity data to a given value.
    :param log: True will result in a log scale display of intensity.
    :param color_scheme: choice of color scheme (given with an int value).
    :return: fig, ax, color_map (pcolor), color_bar
    """
```
