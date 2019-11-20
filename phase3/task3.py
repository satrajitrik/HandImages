from database import Database
from ppr import PageRank


def starter(k, K, seed_images):
    image_vectors = Database().retrieve_many(collection_type="training")
    # print(image_vectors[0:2])
    page_rank = PageRank(image_vectors, k)
    page_rank.generate_graph()
    # print(page_rank.get_graph())
    queue = page_rank.perform_random_walk(seed_images)
    ranked_images = []
    for i in range(K):
        probability, image_name = queue.get()
        probability = -probability
        ranked_images.append((image_name, probability))
    print(ranked_images)
    return ranked_images
    
if __name__ == '__main__':
    starter(k=4, K=10, seed_images=['Hand_0000027', 'Hand_0000757', 'Hand_0000055'])
