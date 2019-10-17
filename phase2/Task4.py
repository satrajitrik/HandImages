import functions

from database import Database
from descriptor import DescriptorType
from labels import Labels
from latentsymantics import LatentSymanticsType


def starter(feature_model, dimension_reduction, k, label_choice, image_id, m):
    label, value, _ = Labels(label_choice).label
    descriptor_type = DescriptorType(feature_model).descriptor_type
    symantics_type = LatentSymanticsType(dimension_reduction).symantics_type

    target_results = Database().retrieve_many(
        descriptor_type, symantics_type, k, label, value
    )
    source_result = Database().retrieve_one(
        image_id, descriptor_type, symantics_type, k, label, value
    )

    print(functions.compare(source_result, target_results, m, descriptor_type))
