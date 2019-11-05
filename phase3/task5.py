import numpy as np

from imageprocessor import ImageProcessor
from lsh import LSH


def starter():
    # Creating a dummy input vector of 100 arrays each having size 10 and values in [-199, 199]
    # input_vector = np.array([np.random.randint(-199, 200, size=10) for _ in range(100)])
    ids, input_vector = ImageProcessor().id_vector_pair

    modified_input_vector = [
        (ids[i], input_vector[i])
        for i in range(len(ids))
    ]

    # Displaying the 5 hashtables
    LSH(5, 5, modified_input_vector).beautify_and_print()
