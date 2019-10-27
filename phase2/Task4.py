import functions

from database import Database
from labels import Labels


def starter(feature_model, dimension_reduction, k, label_choice, image_id, m):
    if not feature_model:
        target_results = Database().retrieve_many(task=3)
        source_result = Database().retrieve_one(image_id, task=3)
        descriptor_type = source_result["descriptor_type"]
    else:
        label, value, _ = Labels(label_choice).label
        filtered_image_ids = [
            item["image_id"]
            for item in Database().retrieve_metadata_with_labels(label, value)
        ]
        _, _ = functions.store_in_db(
            feature_model, dimension_reduction, k, 4, filtered_image_ids, label, value
        )

        target_results = Database().retrieve_many(task=4)
        source_result = Database().retrieve_one(image_id, task=4)
        descriptor_type = source_result["descriptor_type"]

    print(functions.compare(source_result, target_results, m, descriptor_type))
