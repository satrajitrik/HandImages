import math
import numpy as np
import pandas as pd

from collections import defaultdict
from tabulate import tabulate


class HashTable(object):
    def __init__(self, num_of_hashes, input_vector):
        self.num_of_hashes = num_of_hashes
        self.input_vector = input_vector
        self.dimensions = len((input_vector[0])[1])
        self.hash_table = self.__populate_hash_table()

    def __generate_random_vectors(self):
        random_vectors = []
        for _ in range(self.num_of_hashes):
            rv = np.random.normal(0, 1, size=self.dimensions)
            norm = np.linalg.norm(rv)
            random_vectors.append(rv / norm)

        return np.array(random_vectors)

    def __populate_hash_table(self):
        """
        Follows Euclidean family of LSH.
        w value to change with input vectors.
        Hash implementation might change.
        """
        hash_table, w = defaultdict(set), 30
        random_vectors = self.__generate_random_vectors()

        for id, iv in self.input_vector:
            dot_product = np.dot(iv, np.transpose(random_vectors))
            sums = np.add(dot_product, np.random.uniform(0, w)) / w
            hash = "".join(["0" if h < 0 else "1" for h in sums])
            hash_table[hash].add(id)

        return hash_table


class LSH(object):
    def __init__(self, l, k, input_vector):
        self.l = l
        self.k = k
        self.input_vector = input_vector
        self.hash_tables = self.__create_hash_tables()

    def __create_hash_tables(self):
        return [HashTable(self.k, self.input_vector).hash_table for _ in range(self.l)]

    def __beautify_and_print(self, hash_tables):
        """
        Helper function to visualize the l hashtables.
        Only to be used for visualization.
        """
        for hash_table in hash_tables:
            print(
                tabulate(
                    pd.DataFrame(list(hash_table.items())),
                    headers=["Hash", "Image IDs"],
                    showindex=False,
                )
            )

    def get_search_results(self, image_id, show=False):
        hash_tables = self.hash_tables

        if show:
            self.__beautify_and_print(hash_tables)

        search_results = set.union(
            *[
                hash_table[hash]
                for hash_table in hash_tables
                for hash in hash_table
                if image_id in hash_table[hash]
            ]
        )

        return search_results - set([image_id])
