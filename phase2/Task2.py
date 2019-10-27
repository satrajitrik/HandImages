import functions

from database import Database
from descriptor import DescriptorType


def starter(feature_model, dimension_reduction, k, image_id, m):
    if not feature_model:
        target_results = Database().retrieve_many(task=1)
        source_result = Database().retrieve_one(image_id, task=1)
        descriptor_type = source_result["descriptor_type"]
    else:
        _, _ = functions.store_in_db(feature_model, dimension_reduction, k, 2)

        target_results = Database().retrieve_many(task=2)
        source_result = Database().retrieve_one(image_id, task=2)
        descriptor_type = source_result["descriptor_type"]

    print(functions.compare(source_result, target_results, m, descriptor_type))
