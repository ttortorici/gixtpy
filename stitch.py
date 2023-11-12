import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os.path


tif_names = ('1_1_2', '1_1_1', '1_2_2', '1_2_1', '1_3_2', '1_3_1')


def remove_bands(path):
    tifs = __load(path)
    tifs = __remove_artifacts(tifs)

    chopped_top = tifs[:, :514, :].astype(np.float32)
    chopped_bottom = tifs[:, 551:, :].astype(np.float32)

    for ii in range(len(tif_names)):
        row = ii & 1    # (0, 1, 0, 1, 0, 1, ...)
        col = ii // 2   # (0, 0, 1, 1, 2, 2, ...)
        tifffile.imwrite(os.path.join(path, f'r{row}-c{col}.tif'), chopped_top[ii, :, :], imagej=True)
        tifffile.imwrite(os.path.join(path, f'r{row + 2}-c{col}.tif'), chopped_bottom[ii, :, :], imagej=True)
    print("done")


def __load(path):
    filenames = [os.path.join(path, tif_name) for tif_name in tif_names]
    tifs = np.empty((len(filenames), 1065, 1030), dtype=int)
    for ii, f in enumerate(filenames):
        tifs[ii, :, :] = np.array(tifffile.imread(f), dtype=int)
    return tifs


def __remove_artifacts(tifs):
    hot_spots = ((398, 579),
                 (904, 561),
                 (932, 406))
    square_starts = ((255, 255),
                     (255, 513),
                     (255, 771),
                     (808, 255),
                     (808, 513),
                     (808, 771))
    for tt in range(len(tif_names)):
        for z, x in hot_spots:
            sum_to_ave = 0
            for ii in (-1, 0, 1):
                for jj in (-1, 0, 1):
                    if ii or jj:
                        sum_to_ave += tifs[tt, z + jj, x + ii]
            tifs[tt, z, x] = sum_to_ave / 8.
        for z, x in square_starts:
            top = tifs[tt, z-1, x-1:x+5]
            bot = tifs[tt, z+5, x-1:x+5]
            lft = tifs[tt, z:z+4, x-1]
            rgt = tifs[tt, z:z+4, x+5]
            surrounding = np.concatenate((top, bot, lft, rgt))
            # s_ave = np.average(surrounding)
            # s_std = np.std(surrounding)
            # max_val = int(s_std * 3.428268118159367 + 0.17494531140021216)
            replacement = np.random.randint(np.min(surrounding), np.max(surrounding), size=16).reshape((4, 4))
            tifs[tt, z:z+4, x:x+4] = replacement
    tifs[tifs < 0] = 0
    return tifs



def __dezinger(tifs):
    zinger_threshold = np.average(tifs) + np.std(tifs)
    print(np.where(tifs > zinger_threshold))

    zinger_locations = np.where(tifs > zinger_threshold)

    for t, z, x in zip(*zinger_locations):
        sum_to_ave = 0
        for ii in (-1, 0, 1):
            for jj in (-1, 0, 1):
                if ii or jj:
                    sum_to_ave += tifs[t, z + jj, x + ii]
        tifs[t, z, x] = sum_to_ave / 8.
    return tifs


def __test_artifacts(path):
    tifs = __load(path)
    tifs = __remove_artifacts(tifs).astype(np.float32)
    for tt in range(len(tif_names)):
        tifffile.imwrite(f'{tif_names[tt]}.tif', tifs[tt, :, :], imagej=True)


def __test_stitch(path):
    tifs = __load(path)
    tifs = __remove_artifacts(tifs)

    chopped_top = tifs[:, :514, :]
    chopped_bottom = tifs[:, 551:, :]

    full_image = np.zeros((1102, 3070), dtype=np.float32)
    full_image[:514, :1030] += chopped_top[0, :, :]
    full_image[551:1065, :1030] += chopped_bottom[0, :, :]
    full_image[37:551, :1030] += chopped_top[1, :, :]
    full_image[588:, :1030] += chopped_bottom[1, :, :]

    full_image[:514, 1020:2050] += chopped_top[2, :, :]
    full_image[551:1065, 1020:2050] += chopped_bottom[2, :, :]
    full_image[37:551, 1020:2050] += chopped_top[3, :, :]
    full_image[588:, 1020:2050] += chopped_bottom[3, :, :]

    full_image[:514, 2040:] += chopped_top[4, :, :]
    full_image[551:1065, 2040:] += chopped_bottom[4, :, :]
    full_image[37:551, 2040:] += chopped_top[5, :, :]
    full_image[588:, 2040:] += chopped_bottom[5, :, :]

    # for ii in range(tifs.shape[0]):
    tifffile.imwrite(
        'temp_imagej.tif', full_image, imagej=True  # metadata={'axes': 'TZYX', 'fps': 10.0}
    )


    # plt.show()


def test_rand(size):
    x = np.arange(size)
    y = np.zeros(size)
    for ii in range(2, size):
        y[ii] = np.std(np.random.randint(0, ii, size=10000))
    x = x[2:]
    y = y[2:]
    a, b = np.polyfit(x, y, 1)
    plt.scatter(x, y)
    plt.plot(x, a*x+b)
    print(a, b)
    plt.show()


if __name__ == "__main__":
    __test_stitch('C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\TPP Films\\Unkown inclusion (benz, tolu, xyl)\\giwaxs\\TT5mm-02_2023-11-06')
    # test_rand(20)
