import numpy as np

from lsh import LSH


def starter():
    # Creating a dummy input vector of 100 arrays each having size 10 and values in [-10, 10]
    input_vector = np.array([np.random.randint(-10, 11, size=10) for _ in range(100)])

    # Displaying the 5 hashtables
    LSH(5, 5, input_vector).beautify_and_print()
