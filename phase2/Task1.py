import cv2
import numpy as np
import os
from config import Config
from descriptor import Descriptor
from latentsymantics import LatentSymantics


def starter(feature_model, dimension_reduction, k):
    config_object = Config()
    path = config_object.read_path()

    files = os.listdir(path)

    x = []
    for file in files:
        print("Reading file: {}".format(file))
        image = cv2.imread(path + file)

        feature_descriptor = Descriptor(image, feature_model).feature_descriptor
        x.append(feature_descriptor)

    x = np.array(x)

    latent_symantics = LatentSymantics(x, k, dimension_reduction).latent_symantics

    print(latent_symantics)
