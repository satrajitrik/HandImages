import numpy as np

from database import Database
from imageprocessor import ImageProcessor
from lsh import LSH


def starter():
    id_vector_pairs = Database().retrieve_many()

    # Displaying the 5 hashtables
    target_hash_tables = LSH(10, 3, id_vector_pairs).hash_tables

    source_id, source_vector = ImageProcessor(["Hand_0011744"]).id_vector_pair
    source_id_vector_pair = [(source_id[i], source_vector[i]) for i in range(len(source_id))]

    source_hash_tables = LSH(40, 3, source_id_vector_pair).hash_tables

    for hash_table in source_hash_tables:
    	keys = hash_table.keys()
    	for key in keys:
    		for target_hash_table in target_hash_tables:
    			if key in target_hash_table:
    				print(target_hash_table[key])
    			


