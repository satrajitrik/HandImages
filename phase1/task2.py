import cv2
import json
import os
import pickle
import traceback
import sift
import lbp

from bson.binary import Binary
from pymongo import MongoClient


DATABASE_NAME = "mwdb"
COLLECTION_NAME = "hands"
READ_PATH = "/Users/satrajitmaitra/Downloads/Hands_smaller/"
WRITE_PATH = "/Users/satrajitmaitra/Downloads/Hands_features/"

def store_feature_vectors(collection):
	files = os.listdir(READ_PATH)

	for file in files:
		print(file)
		img = cv2.imread(READ_PATH + file)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		feature_vector = {
			"name": file.replace(".jpg", ""),
			"lbp": lbp.lbp(gray),
			"sift": sift.sift(gray)
		}
		with open(WRITE_PATH+file.replace(".jpg", "")+"_fd.json", "w") as fp:
			json.dump(feature_vector, fp, indent=4, sort_keys=True)

		feature_vector["lbp"] = Binary(pickle.dumps(lbp.lbp(gray), protocol=2))
		feature_vector["sift"] = Binary(pickle.dumps(sift.sift(gray), protocol=2))

		collection.insert_one(feature_vector)

def main():
	try:
		connection = MongoClient('mongodb://localhost:27017/')

		print("Creating database in MongoDB named as " + DATABASE_NAME)
		database = connection[DATABASE_NAME]

		print("Creating a collection in " + DATABASE_NAME + " named as " + COLLECTION_NAME)
		collection = database[COLLECTION_NAME]

		store_feature_vectors(collection)

		connection.close()

	except Exception as detail:
	    traceback.print_exc()

if __name__ == '__main__':
	main()
