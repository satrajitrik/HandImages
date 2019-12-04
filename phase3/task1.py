import os

import cv2
from self import self

import functions_phase2
import numpy as np

from config import Config
from database import Database
from descriptor import DescriptorType, Descriptor

from latentsymantics import LatentSymanticsType, LatentSymantics


class task1(object):
    def __init__(self):
        self.x= 0
        # self.p_count_predicted = 0
        # self.dorsal_count =0
        # self.palmer_count =0

    def findlabel(self, feature_model, dimension_reduction, k, image_id):
        descriptor_type = DescriptorType(feature_model).descriptor_type
        symantics_type = LatentSymanticsType(dimension_reduction).symantics_type
        label, value, complementary_value = ("dorsal", 1, 0)
        d_count_predicted = 0
        p_count_predicted = 0

        source = Database().retrieve_one_t1(5,image_id)
        label_targets = Database().retrieve_many_t1( 5, label, value)
        complementary_label_targets = Database().retrieve_many_t1(
            5, label, complementary_value
        )

        label_similarity_info = functions_phase2.compare(source, label_targets, 1, descriptor_type)
        complementary_label_similarity_info = functions_phase2.compare(
            source, complementary_label_targets, 1, descriptor_type
        )

        if label_similarity_info[0][1] > complementary_label_similarity_info[0][1]:
            predicted = "dorsal"
            # d_count_predicted = d_count_predicted + 1
        # Labels(label_choice)._detupleize_label((label, value))

        else:
             predicted = "palmer"
             # p_count_predicted = p_count_predicted + 1
             # Labels(label_choice)._detupleize_label((label, complementary_value))

        print("the image {} is {}".format(image_id, predicted))
        return predicted




    def helper(self,feature_model, dimension_reduction, k):
        unlabelled_path = "C:/Users/himan/OneDrive/Desktop/MWDB/phase3_sample_data/Unlabelled/Set 1/"
        files = os.listdir(unlabelled_path)
        path, pos = Config().read_path(), None
        descriptor_type = DescriptorType(feature_model).descriptor_type
        symantics_type = LatentSymanticsType(dimension_reduction).symantics_type
        label, value, complementary_value = ("dorsal", 1, 0)
        unlabelled_image_feature_vector = []
        unlabelled_image_ids = []


        for i, file in enumerate(files):
            print(file)

            image = cv2.imread("{}{}".format(unlabelled_path, file))
            image_feature_vector = Descriptor(
                image, feature_model, dimension_reduction
            ).feature_descriptor
            unlabelled_image_feature_vector.append(image_feature_vector)
            unlabelled_image_ids.append(file)




        label_filtered_image_ids = [
            item["image_id"]
            for item in Database().retrieve_metadata_with_labels(label, value)
        ]
        complementary_label_filtered_image_ids = [
            item["image_id"]
            for item in Database().retrieve_metadata_with_labels(label, complementary_value)
        ]

        if DescriptorType(feature_model).check_sift():
            label_feature_vector, label_ids, label_pos = functions_phase2.process_files(
                path, feature_model, dimension_reduction, label_filtered_image_ids
            )
            complementary_label_feature_vector, complementary_label_ids, complementary_label_pos = functions_phase2.process_files(
                path,
                feature_model,
                dimension_reduction,
                complementary_label_filtered_image_ids,
            )
            feature_vector = np.concatenate(
                (
                    label_feature_vector,
                    complementary_label_feature_vector,
                    unlabelled_image_feature_vector,
                )
            )
            # pos = label_pos + complementary_label_pos + [image_feature_vector.shape[0]]
        else:
            label_feature_vector, label_ids = functions_phase2.process_files(
                path, feature_model, dimension_reduction, label_filtered_image_ids
            )
            complementary_label_feature_vector, complementary_label_ids = functions_phase2.process_files(
                path,
                feature_model,
                dimension_reduction,
                complementary_label_filtered_image_ids,
            )

            feature_vector = np.concatenate(
                (
                    label_feature_vector,
                    complementary_label_feature_vector,
                    unlabelled_image_feature_vector

                )
            )

        ids = label_ids + complementary_label_ids + unlabelled_image_ids

        _, latent_symantics = LatentSymantics(
            feature_vector, k, dimension_reduction
        ).latent_symantics

        # for i, ids in unlabelled_image_ids:
        #     _, latent_symantics = LatentSymantics(
        #         unlabelled_image_feature_vector[i], k, dimension_reduction
        #     ).latent_symantics

        records = functions_phase2.set_records(
            ids, descriptor_type, symantics_type, k, latent_symantics, pos, 5
        )

        for record in records:

            if record["image_id"] in label_ids:
                record[label] = value
            elif record["image_id"] in complementary_label_ids:
                record[label] = complementary_value
            else:
                continue

        Database().insert_many(records)


    def starter(self, feature_model=1, dimension_reduction=1, k=5):
        dorsal_count =0
        palmer_count = 0
        d_count_predicted = 0
        p_count_predicted = 0

        unlabelled_path = "C:/Users/himan/OneDrive/Desktop/MWDB/phase3_sample_data/Unlabelled/Set 1/"
        files = os.listdir(unlabelled_path)
        # for i, file in enumerate(files):
        #     print(file)
        self.helper(feature_model, dimension_reduction, k)
        for i, file in enumerate(files):
            predicted =  self.findlabel(feature_model, dimension_reduction, k, file)
            imagename = file.replace(".jpg", "")

            if Database().find_label_from_metadata(imagename) == 1:
                dorsal_count = dorsal_count + 1
                if predicted == "dorsal":
                    d_count_predicted = d_count_predicted + 1
            else:
                palmer_count = palmer_count + 1
                if predicted == "palmer":
                    p_count_predicted = p_count_predicted + 1

        print("accuracy for dorsal  {} ".format(d_count_predicted / dorsal_count) * 100 )
        print("accuracy for dorsal  {} ".format(p_count_predicted / palmer_count) * 100 )
