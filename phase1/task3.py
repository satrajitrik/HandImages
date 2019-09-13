from pymongo import MongoClient


DATABASE_NAME = "mwdb"
COLLECTION_NAME = "hands"

def main():
	image_id, model, k = input("Enter image ID, model name (LBP or SIFT) and k: ").split()

	try:
		conn = MongoClient('mongodb://localhost:27017/')

		database = conn[DATABASE_NAME]
		collection = database[COLLECTION_NAME]


		
		conn.close()

	except Exception as detail:
	    traceback.print_exc()

if __name__ == '__main__':
	main()