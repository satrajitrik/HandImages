import functions
import numpy as np
import visualizer

from database import Database
from lsh import LSH


def starter(image_id, m, k, l):
    query_results = Database().retrieve_many()
    id_vector_pairs = [(item["image_id"], item["vector"]) for item in query_results]

    search_results = LSH(l, k, id_vector_pairs).get_search_results(image_id, show=True)

    print(
        "Original dataset size: {} | Reduced search space size: {} | Reduction by {} %".format(
            len(id_vector_pairs),
            len(search_results),
            float(len(id_vector_pairs) - len(search_results))
            * 100
            / len(id_vector_pairs),
        )
    )
    query_results = Database().retrieve_many(list(search_results))
    search_id_vector_pairs = [
        (item["image_id"], item["vector"]) for item in query_results
    ]
    source_vector = Database().retrieve_one(image_id)["vector"]

    similar_images = functions.find_similarity(source_vector, search_id_vector_pairs, m)
    print(similar_images)

    visualizer.visualize_lsh(image_id, similar_images)
