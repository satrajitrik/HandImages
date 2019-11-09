import functions
import numpy as np
import visualizer

from database import Database
from imageprocessor import ImageProcessor
from lsh import LSH


def starter(image_id, m):
    id_vector_pairs = Database().retrieve_many()

    search_results = LSH(6, 6, id_vector_pairs).get_search_results(image_id)

    print(
        "Original dataset size: {} | Reduced search space size: {} | Reduction by {} %".format(
            len(id_vector_pairs),
            len(search_results),
            float(len(id_vector_pairs) - len(search_results))
            * 100
            / len(id_vector_pairs),
        )
    )
    search_id_vector_pairs = Database().retrieve_many(list(search_results))
    source_vector = Database().retrieve_one(image_id)["vector"]

    similar_images = functions.find_similarity(source_vector, search_id_vector_pairs, m)
    print(similar_images)

    visualizer.visualize_lsh(image_id, similar_images)
