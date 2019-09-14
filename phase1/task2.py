import cv2
import json
import os
import pickle
import traceback

import constants
import lbp
import sift

from bson.binary import Binary
from pymongo import MongoClient


def store_feature_vectors(collection, read_path, write_path):
	files = os.listdir(read_path)

	for file in files:
		print(file)
		img = cv2.imread(read_path + file)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		lbp_results = lbp.lbp(gray)
		sift_results = sift.sift(gray)

		feature_vector = {
			"name": file.replace(".jpg", ""),
			"lbp": lbp_results.tolist(),
			"sift": sift_results.tolist()
		}
		with open(write_path+file.replace(".jpg", "")+"_fd.json", "w") as fp:
			json.dump(feature_vector, fp, indent=4, sort_keys=True)

		feature_vector["lbp"] = Binary(pickle.dumps(lbp_results))
		feature_vector["sift"] = Binary(pickle.dumps(sift_results))

		collection.insert_one(feature_vector)

def main():
	constants_dict = constants.read_json()
	db_name = constants_dict["DATABASE_NAME"]
	collection_name = constants_dict["COLLECTION_NAME"]
	read_path = constants_dict["READ_PATH"]
	write_path = constants_dict["WRITE_PATH"]

	try:
		connection = MongoClient("mongodb://localhost:27017/")


		database = connection[db_name]
		collection = database[collection_name]

		store_feature_vectors(collection, read_path, write_path)

		connection.close()

	except Exception as detail:
		traceback.print_exc()

if __name__ == "__main__":
	main()
