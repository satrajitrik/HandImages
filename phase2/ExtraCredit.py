import pandas as pd
import functions
import Visualizer
import numpy as np
from config import Config
from descriptor import DescriptorType
from latentsymantics import LatentSymantics, LatentSymanticsType


def starter(feature_model, dimension_reduction, k,visualizer):
    path, pos = Config().read_path(), None
    descriptor_type = DescriptorType(feature_model).descriptor_type
    if DescriptorType(feature_model).check_sift():
        x, ids, pos = functions.process_files(path, feature_model)
    else:
        x, ids = functions.process_files(path, feature_model)

    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type
    if visualizer == 1:
        _, latent_symantics = LatentSymantics(x, k, dimension_reduction).latent_symantics
        k_th_eigenvector_all = []
        for i in range(k):
            col = latent_symantics[:, i]
            arr = []
            for k, val in enumerate(col):
                arr.append((str(ids[k]+".jpg"), val))
            arr.sort(key=lambda x: x[1], reverse=True)
            k_th_eigenvector_all.append(arr)
            print("Printing term-weight pair for latent Semantic {}:".format(i + 1))
            print(arr)
        k_th_eigenvector_all = pd.DataFrame(k_th_eigenvector_all)
        Visualizer.visualize_data_symantics(k_th_eigenvector_all, symantics_type, descriptor_type)
    elif visualizer == 2:
        latent_symantics, _ = LatentSymantics(x, k, dimension_reduction).latent_symantics
        k_th_eigenvector_all = []
        for j in range(k):
            arr = []
            for i in range(len(ids)):
                arr.append((str(ids[i]+".jpg"), np.dot(x[i], latent_symantics.components_[j])))
                #k_th_eigenvector_all[ids[i]] = np.dot(x[i], latent_symantics[j])
            arr.sort(key=lambda x: x[1], reverse=True)
            k_th_eigenvector_all.append(arr[:1])
            print(arr[0])
        k_th_eigenvector_all = pd.DataFrame(k_th_eigenvector_all)
        Visualizer.visualize_feature_symantics(k_th_eigenvector_all, symantics_type, descriptor_type)

