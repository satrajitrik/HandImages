from database import Database
from descriptor import DescriptorType
from latentsymantics import LatentSymanticsType
from scipy.spatial import distance

"""
	Similarity results not good. Need to come up with something better.
"""


def compare(target, others, m):
    others = [
        {"image_id": item["image_id"], "latent_symantics": item["latent_symantics"]}
        for item in others
        if item["image_id"] != target["image_id"]
    ]
    distances = [
        (
            other["image_id"],
            distance.euclidean(target["latent_symantics"], other["latent_symantics"]),
        )
        for other in others
    ]
    distances = sorted(distances, key=lambda x: x[1])

    return distances[:m]


def starter(feature_model, dimension_reduction, k, image_id, m):
    descriptor_type = DescriptorType(feature_model).descriptor_type
    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type

    query_results = Database().retrieve_many(descriptor_type, symantics_type, k)
    target_result = Database().retrieve_one(
        image_id, descriptor_type, symantics_type, k
    )

    print(compare(target_result, query_results, m))
