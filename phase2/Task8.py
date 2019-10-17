import numpy

from database import Database
from latentsymantics import LatentSymantics
import functions
from sklearn.decomposition import NMF

def starter(k):
    # Get all the metadata from DB
    _,metadata_query_output = Database().retrieve_metadata_with_labels1(None, None)
    """
    Column index - Feature:
    0 - Male
    1 - Female
    2 - Dorsal
    3 - Palmar
    4 - Left
    5 - Right
    6 - With accessories
    7 - Without accessories
    
    Image-Metadata matrix with columns = 8 and rows = # of images
    # """
    image_metadata_matrix = numpy.zeros(shape=(8,len(metadata_query_output)))
    for i in range(len(metadata_query_output)):

        image_metadata_matrix[0][i] = metadata_query_output[i][1]
        image_metadata_matrix[1][i]  = 1 - metadata_query_output[i][1]
        image_metadata_matrix[2][i]  = metadata_query_output[i][2]
        image_metadata_matrix[3][i]  = 1 - metadata_query_output[i][2]
        image_metadata_matrix[4][i]  = metadata_query_output[i][3]
        image_metadata_matrix[5][i]  = 1 - metadata_query_output[i][3]
        image_metadata_matrix[6][i]  = metadata_query_output[i][4]
        image_metadata_matrix[7][i]  = 1 - metadata_query_output[i][4]

    metadataspace = image_metadata_matrix.transpose()

    imagespace_NMF_model,imagespace_latent_symantics = LatentSymantics(image_metadata_matrix, k, choice=3).latent_symantics # [8 X 11k] to [8 X 4] ((11k)d to 4d)
    for index,latent_feature in enumerate(imagespace_NMF_model.components_):
            print("top 50 features for latent_topic #",index)
            print([i for i in latent_feature.argsort()[-50:]])
            print("\n")

    metadataspace_NMF_model,metadataspace_latent_symantics = LatentSymantics(metadataspace, k, choice=3).latent_symantics # [11k X 8] to [11k X 4] (8d to 4d)
    for index,latent_feature in enumerate(metadataspace_NMF_model.components_):
            print("top 50 features for latent_topic #",index)
            print([i for i in latent_feature.argsort()[-50:]])
            print("\n")

