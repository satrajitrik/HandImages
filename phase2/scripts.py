import pandas
from pymongo import MongoClient

from config import Config


def load_metadata_to_db():
    connection = MongoClient(Config().mongo_url())
    database = connection[Config().database_name()]
    collection = database[Config().metadata_collection_name()]
    metadata = pandas.read_csv(Config().metadata_file())
    
    collection.drop()
    for index, row in metadata.iterrows():
        image_name = row['imageName'].replace(".jpg", "")
        male = 1 if row['gender'] == "male" else 0
        dorsal = 1 if "dorsal" in row['aspectOfHand'] else 0
        left_hand = 1 if "left" in row['aspectOfHand'] else 0
        accessories = row['accessories']
        
        collection.insert_one(
            {
                'image_id': image_name,
                'male': male,
                'dorsal': dorsal,
                'left_hand': left_hand,
                'accessories': accessories
            }
        )


if __name__ == '__main__':
    load_metadata_to_db()
