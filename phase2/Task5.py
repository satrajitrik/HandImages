import cv2
import functions
import numpy as np

from config import Config
from database import Database
from descriptor import DescriptorType, Descriptor
from labels import Labels
from latentsymantics import LatentSymanticsType, LatentSymantics


def findlabel(feature_model, dimension_reduction, k, label_choice, image_id):
    descriptor_type = DescriptorType(feature_model).descriptor_type
    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type
    label, value, complementary_value = Labels(label_choice).label

    source = Database().retrieve_one(image_id, 5)
    label_targets = Database().retrieve_many(5, label, value)
    complementary_label_targets = Database().retrieve_many(
        5, label, complementary_value
    )

    label_similarity_info = functions.compare(source, label_targets, 1, descriptor_type)
    complementary_label_similarity_info = functions.compare(
        source, complementary_label_targets, 1, descriptor_type
    )

    if label_similarity_info[0][1] > complementary_label_similarity_info[0][1]:
        predicted = Labels(label_choice)._detupleize_label((label, value))
    else:
        predicted = Labels(label_choice)._detupleize_label((label, complementary_value))

    print(predicted)


def helper(feature_model, dimension_reduction, k, label_choice, image_id):
    path, pos = Config().read_path(), None
    descriptor_type = DescriptorType(feature_model).descriptor_type
    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type
    label, value, complementary_value = Labels(label_choice).label

    image = cv2.imread("{}{}{}".format(Config().read_all_path(), image_id, ".jpg"))
    image_feature_vector = Descriptor(
        image, feature_model, dimension_reduction
    ).feature_descriptor

    label_filtered_image_ids = [
        item["image_id"]
        for item in Database().retrieve_metadata_with_labels(label, value)
    ]
    complementary_label_filtered_image_ids = [
        item["image_id"]
        for item in Database().retrieve_metadata_with_labels(label, complementary_value)
    ]

    if DescriptorType(feature_model).check_sift():
        label_feature_vector, label_ids, label_pos = functions.process_files(
            path, feature_model, dimension_reduction, label_filtered_image_ids
        )
        complementary_label_feature_vector, complementary_label_ids, complementary_label_pos = functions.process_files(
            path,
            feature_model,
            dimension_reduction,
            complementary_label_filtered_image_ids,
        )
        feature_vector = np.concatenate(
            (
                label_feature_vector,
                complementary_label_feature_vector,
                image_feature_vector,
            )
        )
        pos = label_pos + complementary_label_pos + [image_feature_vector.shape[0]]
    else:
        label_feature_vector, label_ids = functions.process_files(
            path, feature_model, dimension_reduction, label_filtered_image_ids
        )
        complementary_label_feature_vector, complementary_label_ids = functions.process_files(
            path,
            feature_model,
            dimension_reduction,
            complementary_label_filtered_image_ids,
        )
        feature_vector = np.concatenate(
            (
                label_feature_vector,
                complementary_label_feature_vector,
                np.array([image_feature_vector]),
            )
        )

    ids = label_ids + complementary_label_ids + [image_id]

    _, latent_symantics = LatentSymantics(
        feature_vector, k, dimension_reduction
    ).latent_symantics

    records = functions.set_records(
        ids, descriptor_type, symantics_type, k, latent_symantics, pos, 5
    )

    for record in records:
        if record["image_id"] == image_id:
            continue
        elif record["image_id"] in label_ids:
            record[label] = value
        elif record["image_id"] in complementary_label_ids:
            record[label] = complementary_value

    Database().insert_many(records)


def starter(feature_model, dimension_reduction, k, label_choice, imageID):
    config_object = Config()

    helper(feature_model, dimension_reduction, k, label_choice, imageID)
    findlabel(feature_model, dimension_reduction, k, label_choice, imageID)
