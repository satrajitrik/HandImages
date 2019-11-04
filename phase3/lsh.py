import numpy as np
import pandas as pd

from collections import defaultdict
from tabulate import tabulate


class HashTable(object):
    def __init__(self, num_of_hashes, input_vector):
        self.num_of_hashes = num_of_hashes
        self.input_vector = input_vector
        self.dimensions = input_vector.shape[1]
        self.hash_table = self.__populate_hash_table()

    def __generate_random_vectors(self):
        random_vectors = []
        for _ in range(self.num_of_hashes):
            rv = np.random.randint(2, size=self.dimensions)
            norm = np.linalg.norm(rv)
            random_vectors.append(rv / norm)

        return np.array(random_vectors)

    def __populate_hash_table(self):
        hash_table = defaultdict(list)
        random_vectors = self.__generate_random_vectors()

        for i, iv in enumerate(self.input_vector):
            dot_product = np.dot(random_vectors, np.transpose(iv))
            hash = "".join(["1" if dp > 0 else "0" for dp in dot_product])
            hash_table[hash].append(i)

        return hash_table


class LSH(object):
    def __init__(self, l, k, input_vector):
        self.l = l
        self.k = k
        self.input_vector = input_vector
        self.hash_tables = self.__create_hash_tables()

    def __create_hash_tables(self):
        return [HashTable(self.k, self.input_vector).hash_table for _ in range(self.l)]

    def beautify_and_print(self):
        """
            Helper function to visualize the l hashtables.
            Only to be used for visualization.
        """
        hash_tables = self.__create_hash_tables()

        for hash_table in hash_tables:
            print(
                tabulate(
                    pd.DataFrame(list(hash_table.items())),
                    headers=["Hash", "Input Array Indices"],
                    showindex=False,
                )
            )
