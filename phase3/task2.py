from descriptor import Descriptor
from pymongo import MongoClient
import traceback
import pandas as pd
import cv2
import math
import numpy as np
from latentsymantics import LatentSymantics
import matplotlib.pyplot as plt
import glob
import os

path = "C:/Users/priya/Documents/images/Phase 3/phase3_sample_data/Labelled/Set1/"
test_path = "C:/Users/priya/Documents/images/Phase 3/phase3_sample_data/Unlabelled/Set 1/*.*"
tolerance = 0.0001


def euclidean(y1, y2):
    squares = 0
    for i in range(len(y1)):
        squares = squares + ((y1[i] - y2[i]) ** 2)
    return math.sqrt(squares)


def clustering(path, c):
    mongo_url = "mongodb://localhost:27017/"
    database_name = "mwdb_phase3"
    lbld_collection_name = "labelled_hands"
    unlbld_collection_name = "unlabelled_hands"
    meta_collection_name = "metadata"
    lbld_csv = "C:/Users/priya/Documents/images/Phase 3/phase3_sample_data/labelled_set1.csv"
    unlabelled_csv = "C:/Users/priya/Documents/images/Phase 3/phase3_sample_data/Unlabelled/unlablled_set1.csv"
    try:
        connection = MongoClient(mongo_url)
        database = connection[database_name]
        lbld_collection = database[lbld_collection_name]
        unlbld_collection = database[unlbld_collection_name]
        meta_collection = database[meta_collection_name]
        # storing labelled images
        df = pd.read_csv(lbld_csv)
        lbld_records = df.to_dict(orient='records')
        lbld_collection.remove()
        lbld_collection.insert_many(lbld_records)

        # storing unlabelled images
        df = pd.read_csv(unlabelled_csv)
        unlbld_records = df.to_dict(orient='records')
        unlbld_collection.remove()
        unlbld_collection.insert_many(unlbld_records)

        ids1, ids2, feature_vector1, feature_vector2, feature_vector3 = [], [], [], [], []
        colors = ['red', 'blue', 'green', 'cyan', 'magenta']
        markers = ['o', '<', 's', '+', 'v', '^', '.', '>', ',', 'd']
        clust_labels = []
        cent_labels = []
        cluster = "Cluster "
        cent = "Centroid "
        for i in range(c):
            clust_labels.append(cluster.join(str(i)))
            cent_labels.append(cent.join(str(i)))
        # extracting features
        # dorsal
        for subject in lbld_collection.find({"aspectOfHand": {"$regex": "dorsal"}}, {"imageName": 1}):
            image_id = subject['imageName']
            img_path = path + image_id
            image = cv2.imread(img_path)
            ids1.append(image_id.replace(".jpg", ""))
            feature_descriptor = Descriptor(image, 1).feature_descriptor
            # normalize features
            features_norm = (feature_descriptor - feature_descriptor.min()) / (
                    feature_descriptor.max() - feature_descriptor.min())
            feature_vector1.append(features_norm)

        _, d_latent_semantics = LatentSymantics(
            np.array(feature_vector1), 2, 1
        ).latent_symantics
        # K means
        centroids, prev_centroids, classes, X, centroid_norm, d_img_classes = [], [], [], [], [], []
        max_iterations = 1
        isOptimal = False
        for i in range(c):
            centroids.append(d_latent_semantics[i])
            prev_centroids.append(d_latent_semantics[i])
        while not isOptimal and max_iterations < 501:
            d_distances = []
            classes = []
            d_img_classes = []
            for i in range(c):
                classes.append([])
                d_img_classes.append([])
            # Calculating clusters for each feature
            for i in range(d_latent_semantics.shape[0]):
                features = d_latent_semantics[i]
                d_distances = [euclidean(features, centroid) for centroid in centroids]
                classification = d_distances.index(min(d_distances))
                classes[classification].append(features)
                d_img_classes[classification].append(ids1[i])
            # Recalculating centroids
            for i in range(len(classes)):
                centroids[i] = np.mean(classes[i], axis=0)
            isOptimal = True
            for i in range(len(centroids)):
                if sum((centroids[i] - prev_centroids[i]) / prev_centroids[i] * 100.0) > tolerance:
                    isOptimal = False
                    break
                prev_centroids[i] = centroids[i]
            max_iterations += 1
        # # Visualize clusters -- takes longer time to show so commented
        # for i in range(c):
        #     plt.scatter(centroids[i][0], centroids[i][1], s=300, c="black", marker="x", label=cent_labels[i])
        #     for features in classes[i]:
        #         plt.scatter(features[0], features[1], color=colors[i], s=30, marker=markers[i], label=clust_labels[i])
        # plt.show()
        print "Dorsal CLusters: "
        for i in range(len(d_img_classes)):
            print ("Cluster %d: " % i)
            print d_img_classes[i]
        # ---------------------------------------------------------------------------------------------------------------------
        # extracting features
        # palmar
        for subject in lbld_collection.find({"aspectOfHand": {"$regex": "palmar"}}, {"imageName": 1}):
            image_id = subject['imageName']
            img_path = path + image_id
            image = cv2.imread(img_path)
            ids2.append(image_id.replace(".jpg", ""));
            # normalize features
            feature_descriptor = Descriptor(image, 1).feature_descriptor
            features_norm = (feature_descriptor - feature_descriptor.min()) / (
                    feature_descriptor.max() - feature_descriptor.min())
            feature_vector2.append(features_norm)
        _, p_latent_semantics = LatentSymantics(
            np.array(feature_vector2), 2, 1
        ).latent_symantics
        # K means
        p_centroids, p_prev_centroids, p_classes, p_X, p_centroid_norm, p_img_classes = [], [], [], [], [], []
        p_max_iterations = 1
        p_isOptimal = False
        for i in range(c):
            p_centroids.append(p_latent_semantics[i])
            p_prev_centroids.append(p_latent_semantics[i])
            p_classes.append([])
            p_img_classes.append([])
        while not p_isOptimal and p_max_iterations < 501:
            p_distances = []
            p_classes = []
            p_img_classes = []
            for i in range(c):
                p_classes.append([])
                p_img_classes.append([])
            # Calculating clusters for each feature
            for i in range(p_latent_semantics.shape[0]):
                features = p_latent_semantics[i]
                p_distances = [euclidean(features, centroid) for centroid in p_centroids]
                classification = p_distances.index(min(p_distances))
                p_classes[classification].append(features)
                p_img_classes[classification].append(ids2[i])
            # Recalculating centroids
            for i in range(len(p_classes)):
                p_centroids[i] = np.mean(p_classes[i], axis=0)
            p_isOptimal = True
            for i in range(len(p_centroids)):
                if sum((p_centroids[i] - p_prev_centroids[i]) / p_prev_centroids[i] * 100.0) > tolerance:
                    p_isOptimal = False
                    break
                p_prev_centroids[i] = p_centroids[i]
            p_max_iterations += 1

        # # Visualize clusters -- takes longer time to show so commented
        # for i in range(c):
        #     plt.scatter(p_centroids[i][0], p_centroids[i][1], s=130, marker="x")
        #     for features in p_classes[i]:
        #         plt.scatter(features[0], features[1], color=colors[i], s=30, marker=markers[i])
        # plt.show()
        print "Palmar CLusters: "
        for i in range(len(p_img_classes)):
            print ("Cluster %d" % i)
            print p_img_classes[i]
        # ----------------------------------------------------------------------------------------------------------------------
        # Classification
        # mean_dorsal = np.mean(centroids, axis=0)
        # mean_palmar = np.mean(p_centroids, axis=0)
        image_name = []
        dorsal_cnt = 0
        palmar_cnt = 0
        d_cnt = 0
        p_cnt = 0
        for image_path in glob.glob(test_path):
            image = cv2.imread(image_path)
            # get filename
            image_name.append(os.path.basename(image_path))
            feature_descriptor = Descriptor(image, 1).feature_descriptor
            # normalize features
            features_norm = (feature_descriptor - feature_descriptor.min()) / (
                    feature_descriptor.max() - feature_descriptor.min())
            feature_vector3.append(features_norm)
        _, latent_semantics = LatentSymantics(np.array(feature_vector3), 2, 1).latent_symantics
        for i in range(len(latent_semantics)):
            ddistances = [euclidean(latent_semantics[i], centroid) for centroid in centroids]
            pdistances = [euclidean(latent_semantics[i], centroid) for centroid in p_centroids]

            subject_img = unlbld_collection.find_one({"imageName": image_name[i]}, {"aspectOfHand": 1})
            if "dorsal" in subject_img['aspectOfHand']:
                d_cnt += 1
            else:
                p_cnt += 1
            if min(ddistances) < min(pdistances):
                if "dorsal" in subject_img['aspectOfHand']:
                    dorsal_cnt += 1
                print ("Image ID: %s, %s" % (image_name[i], "dorsal"))
            else:
                if "palmar" in subject_img['aspectOfHand']:
                    palmar_cnt += 1
                print ("Image ID: %s, %s" % (image_name[i], "palmar"))
        print ("Dorsal Accuracy %d" % ((dorsal_cnt*100)/d_cnt))
        print ("Palmar Accuracy %d" % ((palmar_cnt*100)/p_cnt))

    except Exception as e:
        traceback.print_exc()
        print("Connection refused... ")


if __name__ == '__main__':
    clustering(path, 5)
