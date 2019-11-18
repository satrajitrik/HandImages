from database import Database
from ppr import PageRank


def starter(k, K, seed_images):
    image_vectors = Database().retrieve_many(collection_type="training")
    # print(image_vectors[0:2])
    page_rank = PageRank(image_vectors[:6], 3)
    print(page_rank.get_graph())
    print(page_rank.perform_random_walk([1, 4]))
    
if __name__ == '__main__':
    starter()
