import numpy

from database import Database
from latentsymantics import LatentSymantics
import functions

def starter(k):
    # Get all the metadata from DB
    metadata_query_output = Database().retrieve_metadata_with_labels(None, None)
    
    """
    Column index - Feature:
    0 - Left
    1 - Right
    2 - Dorsal
    3 - Palmar
    4 - With accessories
    5 - Without accessories
    6 - Male
    7 - Female
    
    Image-Metadata matrix with columns = 8 and rows = # of images
    """
    image_metadata_matrix = numpy.zeros((metadata_query_output.count(), 8))
    for index, row in enumerate(image_metadata_matrix):
        image_metadata_matrix[index][0] = row['left_hand']
        image_metadata_matrix[index][1] = 1 - row['left_hand']
        image_metadata_matrix[index][2] = row['dorsal']
        image_metadata_matrix[index][3] = 1 - row['dorsal']
        image_metadata_matrix[index][4] = row['accessories']
        image_metadata_matrix[index][5] = 1 - row['accessories']
        image_metadata_matrix[index][6] = row['male']
        image_metadata_matrix[index][7] = 1 - row['male']
    
    image_metadata_nmf_semantics = LatentSymantics(image_metadata_matrix, k, choice=3).latent_symantics     # Choice 3 is NMF

    print()
    
