import numpy

import functions, task5, visualizer

from database import Database
from ppr import PageRank


def get_feedback(similar_images):
    feedback = []
    for id, _ in similar_images:
        response = input("Is {} relevant? y/n ".format(id))
        feedback.append((id, response))

    return feedback


def probablistic_feedback(image_id, all_images):
    relevant_images = []
    relevant_list, irrelevant_list = Database().retrieve_feedback(image_id)

    for id, _ in all_images:
        """
        Comparing the priors based on Naive Bayes formula. P(B|A) = P(A|B).P(B) / P(A)
        We only need to calculate P(A|B) for comparison
        """
        relevance_prior = relevant_list.count(id) / len(relevant_list)
        irrelevance_prior = irrelevant_list.count(id) / len(irrelevant_list)

        if relevance_prior >= irrelevance_prior:
            relevant_images.append((id, relevance_prior))

    return relevant_images


def ppr_based_feedback(page_rank, image_id):
    output_images = []
    relevant_images, irrelevant_images = Database().retrieve_feedback(image_id=image_id)
    page_rank.s = numpy.ones(page_rank.node_count)
    all_images = page_rank.all_image_names
    for image in relevant_images:
        page_rank.s[all_images.index(image)] = 2
    for image in irrelevant_images:
        page_rank.s[all_images.index(image)] = 0
    
    # Column normalize to one
    if sum(page_rank.s):
        page_rank.s = page_rank.s / sum(page_rank.s)
    queue = page_rank.perform_random_walk()
    while not queue.empty():
        probability, image = queue.get()
        probability = -probability
        if not probability and image not in irrelevant_images:
            output_images.append((image, probability))
    return output_images


def init_ppr(all_images):
    image_vectors = Database().retrieve_many([image[0] for image in set(all_images)])
    page_rank = PageRank(k=8, labelled_images=image_vectors)
    page_rank.generate_graph()
    return page_rank


def feedback_loop(image_id, similar_images, all_images, m, algorithm):
    response = input("Satisfied with search results? y/n ")

    if response == "y":
        return
    
    if algorithm == 3:
        page_rank = init_ppr(all_images)
    feedback = get_feedback(similar_images)

    while 1:
        Database().store_feedback(image_id, feedback)
        if algorithm == 1:
            pass
        elif algorithm == 2:
            pass
        elif algorithm == 3:
            similar_images = sorted(
                ppr_based_feedback(page_rank=page_rank, image_id=image_id),
                key=lambda x: x[1],
                reverse=True,
            )[:m]
        elif algorithm == 4:
            similar_images = sorted(
                probablistic_feedback(image_id, all_images),
                key=lambda x: x[1],
                reverse=True,
            )[:m]
        visualizer.visualize_lsh(image_id, similar_images)

        response = input("Satisfied with search results? y/n ")
        if response == "y":
            break

        feedback = get_feedback(similar_images)

    return


def starter(image_id, m, k, l, algorithm):
    similar_images, all_images = task5.starter(image_id, m, k, l)

    feedback_loop(image_id, similar_images, all_images, m, algorithm)
