from database import Database
from imageprocessor import ImageProcessor


def preprocess_images():
	ids, input_vector = ImageProcessor().id_vector_pair
	records = [
	    {"id": ids[i], "vector": input_vector[i].tolist()}
	    for i in range(len(ids))
	]

	Database().insert_many(records)


if __name__ == '__main__':
	preprocess_images()