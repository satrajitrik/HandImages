import visualizer
from database import Database
from ppr import PageRank


def starter(k, K, seed_images):
    image_vectors = Database().retrieve_many(collection_type="training")
    page_rank = PageRank(k, image_vectors)
    page_rank.generate_graph()
    queue = page_rank.perform_random_walk(seed_images)
    ranked_images = []
    for i in range(K):
        probability, image_name = queue.get()
        probability = -probability
        ranked_images.append((image_name, probability))
    print(ranked_images)
    visualizer.visualize_task3(ranked_images)
    

if __name__ == '__main__':
    starter(k=4, K=15, seed_images=['Hand_0000002', 'Hand_0006340', 'Hand_0006336'])
