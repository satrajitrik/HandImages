import cv2
import numpy as np
import os

from scipy.stats import skew
from sklearn.decomposition import TruncatedSVD

from config import Config


class ImageProcessor(object):
    def __init__(self, filtered_image_ids = None):
        self.read_path = Config().read_all_path() if filtered_image_ids else Config().read_path()
        self.filtered_image_ids = filtered_image_ids
        self.id_vector_pair = self.__process_files()

    def __process_files(self):
        files = os.listdir(self.read_path)

        ids, x = [], []
        for file in files:
            if not self.filtered_image_ids or (
                self.filtered_image_ids and file.replace(".jpg", "") in self.filtered_image_ids
            ):
                print("Reading file: {}".format(file))
                image = cv2.imread("{}{}".format(self.read_path, file))

                feature_descriptor = self.__color_moments(image)

                ids.append(file.replace(".jpg", ""))
                x.append(feature_descriptor)

        return ids, x

    def __color_moments(self, image):
        img_out = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Saving height, width, and channel  of a given image
        y_len, x_len, channel = img_out.shape

        y, u, v = cv2.split(img_out)
        # Defining needed variables to store mean,deviation, skew of different channels of an image

        meanOfY, meanofU, meanofV = [], [], []
        sdOfY, sdofU, sdofV = [], [], []
        skewOfY, skewofU, skewofV = [], [], []

        for i in range(0, y_len, 100):

            for j in range(0, x_len, 100):
                # Slicing image into 100*100 matrix
                # using numpy package to determine mean and deviation of sub blocks
                meanimg = np.nanmean(
                    img_out[i : i + 100, j : j + 100],
                    axis=tuple(range(img_out[i : i + 100, j : j + 100].ndim - 1)),
                )
                deviationimg = np.std(
                    img_out[i : i + 100, j : j + 100],
                    axis=tuple(range(img_out[i : i + 100, j : j + 100].ndim - 1)),
                )

                arr_y = y[i : i + 100, j : j + 100]
                arr_u = u[i : i + 100, j : j + 100]
                arr_v = v[i : i + 100, j : j + 100]

                # appending mean of each color channel of every 100*100 sub matrix
                meanOfY.append(meanimg[0])
                meanofU.append(meanimg[1])
                meanofV.append(meanimg[2])

                # appending standard deviation of each color channel aof every 100*100 sub matrix
                sdOfY.append(deviationimg[0])
                sdofU.append(deviationimg[1])
                sdofV.append(deviationimg[2])

                # appending Skewness of each color channel aof every 100*100 sub matrix
                min_skew = abs(
                    np.nanmin(
                        [
                            skew(arr_y.flatten()),
                            skew(arr_u.flatten()),
                            skew(arr_v.flatten()),
                        ]
                    )
                )

                skewOfY.append(skew(arr_y.flatten()))
                skewofU.append(skew(arr_u.flatten()))
                skewofV.append(skew(arr_v.flatten()))

        color_moment_feature_vector = np.concatenate(
            [meanOfY, meanofU, meanofV, sdOfY, sdofU, sdofV, skewOfY, skewofU, skewofV],
            axis=0,
        )
        return color_moment_feature_vector
